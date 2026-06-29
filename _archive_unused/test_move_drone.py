import time
import airsim

client = airsim.MultirotorClient()
client.confirmConnection()
client.simPause(False)

vehicles = client.listVehicles()
print("Vehicles:", vehicles)

if "Chaser" in vehicles:
    vehicle = "Chaser"
elif len(vehicles) > 0:
    vehicle = vehicles[0]
else:
    vehicle = ""

print("Using vehicle:", vehicle)

client.enableApiControl(True, vehicle_name=vehicle)
client.armDisarm(True, vehicle_name=vehicle)

print("Takeoff...")
client.takeoffAsync(timeout_sec=10, vehicle_name=vehicle).join()

print("Go to altitude z=-8...")
client.moveToZAsync(-8, 3, vehicle_name=vehicle).join()
time.sleep(1)

print("Move forward in world X direction...")
client.moveByVelocityAsync(5, 0, 0, 3, vehicle_name=vehicle).join()

print("Hover...")
client.hoverAsync(vehicle_name=vehicle).join()

state = client.getMultirotorState(vehicle_name=vehicle)
pos = state.kinematics_estimated.position
print(f"FINAL POS: x={pos.x_val:.2f}, y={pos.y_val:.2f}, z={pos.z_val:.2f}")

print("Done.")
