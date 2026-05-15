"""Compatibility wrapper for basketball_sim.serving.world_model_server."""

from basketball_sim.serving.world_model_server import *  # noqa: F401,F403
from basketball_sim.serving.world_model_server import main


if __name__ == "__main__":
    main()

