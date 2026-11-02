/* LD_PRELOAD shim: cap the beamng_ros2 bridge's scalar publish rate without
 * patching the game library.
 *
 * The unpatched bridge publishes its scalar sensors (state/electrics/time/
 * damage) at the 2 kHz publish-loop rate. That pins a core in the game AND
 * floods the env's rclpy spin thread with ~6k msg/s, which together drag the
 * game's physics tick rate down to a few percent of real time (~2 s/step in
 * the hybrid env vs ~40 ms with the rate limited).
 *
 * This shim interposes rcl_publish (the single C entry point every rclcpp
 * publish goes through) and drops frames that arrive sooner than the cap
 * interval for each publisher. The bridge's 2 kHz loop keeps spinning for
 * low-latency control (spin_some); only the sensor publishes are throttled.
 * Dropped frames return RCL_RET_OK, so the bridge sees a normal publish.
 *
 * Env vars:
 *   BNG_ROS2_PRELOAD_HZ      cap in Hz (default 120, matching the game's
 *                            render-loop rate of 120 fps, so the sensor
 *                            stream stays as fresh as the main loop allows;
 *                            0 disables the cap)
 *   BNG_ROS2_PRELOAD_EXCLUDE comma-separated topic substrings exempt from the
 *                            cap (default: none)
 *
 * Build (see build.sh):
 *   cc -O2 -shared -fPIC -o libbng_ros2_preload.so bng_ros2_preload.c -ldl
 *
 * Export (see drift-ai/env-drift-tools):
 *   export LD_PRELOAD=.../preload/libbng_ros2_preload.so
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int rcl_ret_t;
typedef struct rcl_publisher_s rcl_publisher_t;
typedef struct rmw_publisher_allocation_s rmw_publisher_allocation_t;

typedef rcl_ret_t (*rcl_publish_fn)(const rcl_publisher_t *, const void *,
                                    rmw_publisher_allocation_t *);
typedef const char *(*rcl_topic_name_fn)(const rcl_publisher_t *);

#define MAX_PUBS 64
#define MAX_EXCLUDES 16
#define REPORT_PERIOD_NS 5000000000LL /* 5 s */

typedef struct {
    const rcl_publisher_t * pub;
    char topic[192];
    int excluded;
    int64_t last_ns;
    int64_t n_pub;
    int64_t n_drop;
} PubState;

static PubState g_pubs[MAX_PUBS];
static int g_npubs = 0;
static int64_t g_interval_ns = 0; /* 0 = cap disabled */
static char g_excludes[MAX_EXCLUDES][64];
static int g_nexcludes = 0;
static int g_init = 0;
static int64_t g_start_ns = 0;
static int64_t g_last_report_ns = 0;

static rcl_publish_fn g_real_publish = NULL;
static rcl_topic_name_fn g_real_topic_name = NULL;

static int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

/* Resolve the real rcl_publish. dlsym(RTLD_NEXT) only sees the global
 * symbol scope; rclpy (Python extensions) and the game's bridge plugin load
 * librcl.so RTLD_LOCAL, so fall back to finding the loaded librcl.so in
 * /proc/self/maps and dlopen(RTLD_NOLOAD) it for a direct handle. */
static void resolve_real_rcl(void) {
    g_real_publish =
        (rcl_publish_fn)dlsym(RTLD_NEXT, "rcl_publish");
    g_real_topic_name =
        (rcl_topic_name_fn)dlsym(RTLD_NEXT, "rcl_publisher_get_topic_name");
    if (g_real_publish) return;

    FILE *maps = fopen("/proc/self/maps", "r");
    if (!maps) return;
    char line[512];
    while (fgets(line, sizeof(line), maps)) {
        char *hit = strstr(line, "librcl.so");
        if (!hit) continue;
        char *start = hit;
        while (start > line && *(start - 1) != ' ') start--;
        char *end = line + strlen(line);
        while (end > start && (end[-1] == '\n' || end[-1] == ' ')) end--;
        *end = '\0';
        void *h = dlopen(start, RTLD_NOLOAD | RTLD_NOW);
        if (!h) continue;
        g_real_publish = (rcl_publish_fn)dlsym(h, "rcl_publish");
        g_real_topic_name =
            (rcl_topic_name_fn)dlsym(h, "rcl_publisher_get_topic_name");
        break;
    }
    fclose(maps);
}

static void init_once(void) {
    if (g_init) return;
    g_init = 1;
    resolve_real_rcl();

    const char *hz = getenv("BNG_ROS2_PRELOAD_HZ");
    const double v = hz ? atof(hz) : 120.0;
    if (v > 0.0) g_interval_ns = (int64_t)(1000000000.0 / v);

    const char *ex = getenv("BNG_ROS2_PRELOAD_EXCLUDE");
    if (ex && ex[0]) {
        const char *p = ex;
        while (*p && g_nexcludes < MAX_EXCLUDES) {
            while (*p == ',' || *p == ' ') p++;
            if (!*p) break;
            const char *q = p;
            while (*q && *q != ',' && *q != ' ') q++;
            size_t n = (size_t)(q - p);
            if (n >= sizeof(g_excludes[0])) n = sizeof(g_excludes[0]) - 1;
            memcpy(g_excludes[g_nexcludes], p, n);
            g_excludes[g_nexcludes][n] = '\0';
            g_nexcludes++;
            p = q;
        }
    }

    g_start_ns = now_ns();
    g_last_report_ns = g_start_ns;
    fprintf(stderr,
            "[bng-ros2-preload] rcl_publish: %s; cap: %s (interval %lld ns), "
            "%d exclude pattern(s)\n",
            g_real_publish ? "found" : "NOT FOUND (publishes pass through)",
            g_interval_ns ? "on" : "off", (long long)g_interval_ns,
            g_nexcludes);
}

static int is_excluded(const char *topic) {
    for (int i = 0; i < g_nexcludes; i++) {
        if (g_excludes[i][0] && strstr(topic, g_excludes[i])) return 1;
    }
    return 0;
}

rcl_ret_t rcl_publish(const rcl_publisher_t * publisher,
                      const void * ros_message,
                      rmw_publisher_allocation_t * allocation) {
    if (!g_init) init_once();
    if (!g_real_publish) return 0; /* no rcl in this process */

    if (g_interval_ns > 0 && publisher) {
        PubState *ps = NULL;
        for (int i = 0; i < g_npubs; i++) {
            if (g_pubs[i].pub == publisher) { ps = &g_pubs[i]; break; }
        }
        if (!ps && g_npubs < MAX_PUBS) {
            ps = &g_pubs[g_npubs++];
            memset(ps, 0, sizeof(*ps));
            ps->pub = publisher;
            ps->last_ns = 0; /* first frame always passes */
            if (g_real_topic_name) {
                const char *t = g_real_topic_name(publisher);
                if (t) {
                    strncpy(ps->topic, t, sizeof(ps->topic) - 1);
                    ps->excluded = is_excluded(ps->topic);
                }
            }
        }
        if (ps && !ps->excluded) {
            const int64_t now = now_ns();
            if (now - ps->last_ns < g_interval_ns) {
                ps->n_drop++;
                return 0; /* RCL_RET_OK: drop the frame */
            }
            ps->last_ns = now;
            ps->n_pub++;
        }
    }

    const int64_t now = now_ns();
    if (now - g_last_report_ns >= REPORT_PERIOD_NS) {
        g_last_report_ns = now;
        fprintf(stderr, "[bng-ros2-preload] %llds up: ",
                (long long)((now - g_start_ns) / 1000000000LL));
        for (int i = 0; i < g_npubs; i++) {
            fprintf(stderr, "%s(pub=%lld drop=%lld) ", g_pubs[i].topic,
                    (long long)g_pubs[i].n_pub, (long long)g_pubs[i].n_drop);
        }
        fprintf(stderr, "\n");
    }

    return g_real_publish(publisher, ros_message, allocation);
}
