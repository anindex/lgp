from qibullet import SimulationManager

if __name__ == "__main__":
    simulation_manager = SimulationManager()
    client_id = simulation_manager.launchSimulation(gui=True)
    pepper = simulation_manager.spawnPepper(
        client_id,
        translation=[0, 0, 0],
        quaternion=[0, 0, 0, 1],
        spawn_ground_plane=True)
      