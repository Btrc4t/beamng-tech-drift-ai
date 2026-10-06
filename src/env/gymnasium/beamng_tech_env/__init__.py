from gymnasium.envs.registration import register

register(
    id="beamng_tech_env/BeamNGTech-v0.39",
    entry_point="beamng_tech_env.envs:BeamNGTechEnv",
    nondeterministic=True,
)
