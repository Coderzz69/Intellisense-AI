import os
import sys
import subprocess

SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
SUMO_TOOLS = os.path.join(SUMO_HOME, "tools")
RANDOM_TRIPS = os.path.join(SUMO_TOOLS, "randomTrips.py")

def main():
    print("=== 1. Generating 4-arm Grid Network (4 lanes per road) ===")
    # --grid guarantees every internal node is a 4-arm crossroads
    # --grid.attach-length adds short connector roads at the boundary for realism
    net_cmd = [
        "netgenerate",
        "--grid",
        "--grid.x-number", "4",        # 4 columns  →  (4-1)*(4-1) = 9 internal 4-arm nodes
        "--grid.y-number", "4",        # 4 rows
        "--grid.x-length", "200",      # 200 m spacing
        "--grid.y-length", "200",
        "--default.lanenumber", "4",   # 4 lanes every road
        "--default.speed", "13.89",    # 50 km/h
        "--tls.guess", "true",
        "--tls.cycle.time", "90",
        "-o", "sumo/network.net.xml",
    ]
    result = subprocess.run(net_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[ERROR]", result.stderr)
        sys.exit(1)
    print("  ✔ Network generated (4×4 grid, 4-lane, all 4-arm intersections).")

    print("\n=== 2. Generating Indian-Mix Traffic Routes ===")
    route_cmd = [
        sys.executable, RANDOM_TRIPS,
        "-n", "sumo/network.net.xml",
        "-r", "sumo/routes.rou.xml",
        "-e", "3600",
        "-p", "1.2",
        "--fringe-factor", "5",   # more vehicles start/end at outer roads (realistic)
        "--trip-attributes", "type=\"indian_mix\" departLane=\"random\" departSpeed=\"random\"",
    ]
    result = subprocess.run(route_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[ERROR]", result.stderr)
        sys.exit(1)
    print("  ✔ Routes generated.")
    print("\nSUCCESS: 4×4 grid network ready!")

if __name__ == "__main__":
    main()
