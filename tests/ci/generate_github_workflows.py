import os
from pathlib import Path

def main():
    workflows_dir = Path(".github/workflows")
    for template_path in workflows_dir.glob("*.j2"):
        with open(template_path, "r") as f:
            template_content = f.read()

        yaml_path = Path(str(template_path).replace(".j2", ""))
        with open(yaml_path, "w") as f:
            f.write("# This file is auto-generated from a template. Do not edit manually.\n\n")
            f.write(template_content)

        print(f"Generated {yaml_path} from {template_path}")

if __name__ == "__main__":
    main()
