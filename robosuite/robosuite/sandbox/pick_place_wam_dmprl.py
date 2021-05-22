import argparse

import robosuite as suite

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="PickPlaceMilk")
    parser.add_argument("--robots", type=str, default="WAM", help="Which robot(s) to use in the env")
    parser.add_argument("--config", type=str, default="single-arm-opposed",
                        help="Specified environment configuration if necessary")
    parser.add_argument("--render", action="store_true", help="Render the scene in Mujoco")
    parser.add_argument("--pi2-rollouts", type=int, help="No. of rollouts per PI2 update")
    parser.add_argument('--pi2-iterations', type=int, help="No. of iteration for the PI2 algorithm")
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--controller", type=str, default="osc", help="Choice of controller. Can be 'ik' or 'osc'")
    args = parser.parse_args()

    # Load demonstrated trajectories

    # Train on the demonstrated trajectories using the desired method, to create the movement primitives

    # Create the Pick-and-Place environment with the Milk Carton
    env = suite.make(
        args.environment,
        robots=args.robot,
        has_renderer=args.render,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        gripper_visualizations=True,
        reward_shaping=True,
        control_freq=20
    )

    # Start training loop
    for iter in range(args.pi2_iterations):
        obs = env.reset()

        # Rollout trajectories for PI2 updates
        for rollout in range(args.pi2_rollouts):
            # Add noise to the DMP parameters

            # Rollout the trajectory and save it to the buffer
            pass

        # Using all rollouts and the "perturbed" parameters, update the DMP weights