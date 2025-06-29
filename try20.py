import yaml
import subprocess
import base64
from pathlib import Path
import toml
import json
from collections import OrderedDict
import sys
import time
import os
import re


# Constants
NAMESPACE = "madhu123"
USER_INPUT_PATH = "/home/pcadmin/cloudai/nim-deploy/kserve/scripts/user_input.toml"


def load_input_config(input_filename):
    if not os.path.exists(input_filename):
        raise FileNotFoundError(f"Input TOML file '{input_filename}' not found.")
    
    config = toml.load(input_filename)
    test = config.get("test", {})
    model_name = test.get("model_name")
    concurrency_levels = test.get("concurrency_levels")
    input_seq_len = test.get("input_sequence_length")
    output_seq_len = test.get("output_sequence_length")
    num_nodes = test.get("num_nodes")
    url = test.get("url")
    measurement_interval = test.get("measurement_interval")
    export_file_pattern = test.get("export_file_pattern")

    if not isinstance(concurrency_levels, list) or not all(isinstance(x, int) for x in concurrency_levels):
        raise ValueError("Concurrency levels must be a list of integers.")

    return {
        "model_name": model_name,
        "concurrency_levels": concurrency_levels,
        "input_seq_len": input_seq_len,
        "output_seq_len": output_seq_len,
        "num_nodes": num_nodes,
        "url": url,
        "measurement_interval": measurement_interval,
        "export_file_pattern": export_file_pattern
    }


def create_tests(config, filename="test_config.toml"):
    toml_data = {"tests": []}
    for i, concurrency in enumerate(config["concurrency_levels"], start=1):
        test_entry = OrderedDict()
        test_entry["id"] = f"Test{i}"
        test_entry["model_name"] = config["model_name"]
        test_entry["concurrency_level"] = concurrency
        test_entry["input_sequence_length"] = config["input_seq_len"]
        test_entry["output_sequence_length"] = config["output_seq_len"]
        test_entry["num_nodes"] = config["num_nodes"]
        test_entry["url"] = config["url"]
        test_entry["measurement_interval"] = config["measurement_interval"]
        test_entry["export_file_pattern"] = config["export_file_pattern"]
        toml_data["tests"].append(test_entry)

    with open(filename, "w") as f:
        toml.dump(toml_data, f)

    print(f"\n TOML file '{filename}' created with {len(config['concurrency_levels'])} tests.")


def run_scheduler(filename="test_config.toml"):
    try:
        config = toml.load(filename)
        tests = config.get("tests", [])
        print(f"\n Running {len(tests)} tests:")
        for i, test in enumerate(tests, start=1):
            print(f"\n Test {i}:")
            print(json.dumps(test, indent=3))
    except toml.TomlDecodeError as e:
        print(f"\n TOML syntax error: {e}")
    except Exception as ex:
        print(f"\n Failed to run scheduler: {ex}")



def export_env_vars(api_keys):
    ngc_api_key = api_keys.get("ngc_api_key")
    ngc_token = api_keys.get("ngc_token")
    hf_token = api_keys.get("hugging_face_token")

    if ngc_api_key:
        os.environ["NGC_API_KEY"] = ngc_api_key
        print(f'export NGC_API_KEY="{ngc_api_key}"')
    else:
        print(" Warning: 'ngc_api_key' missing in [api_keys]")

    if ngc_token:
        os.environ["NGC_TOKEN"] = ngc_token
        print(f'export NGC_TOKEN="{ngc_token}"')

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        print(f'export HF_TOKEN="{hf_token}"')
    else:
        print(" Warning: 'hugging_face_token' missing in [api_keys]")

    return ngc_api_key, ngc_token, hf_token

def update_pvc_yaml(pvc_config):
    try:
        with open(pvc_config["pvc_yaml_path"], 'r') as f:
            pvc = yaml.safe_load(f)
    except FileNotFoundError:
        sys.exit(f" PVC YAML file not found: {pvc_config['pvc_yaml_path']}")
    except Exception as e:
        sys.exit(f" Failed to read PVC YAML: {e}")

    if pvc is None:
        sys.exit(f" PVC YAML file is empty: {pvc_config['pvc_yaml_path']}")

    if 'spec' not in pvc:
        sys.exit(f" PVC YAML missing 'spec' section: {pvc_config['pvc_yaml_path']}")

    pvc['spec']['storageClassName'] = pvc_config['storage_class']
    pvc['spec']['resources']['requests']['storage'] = pvc_config['storage_size']

    try:
        with open(pvc_config["pvc_yaml_path"], 'w') as f:
            yaml.dump(pvc, f)
        print(" PVC YAML updated successfully.")
    except Exception as e:
        sys.exit(f" Failed to write PVC YAML: {e}")

    # Apply updated PVC YAML
    try:
        subprocess.run(["kubectl", "apply", "-n", NAMESPACE, "-f", pvc_config["pvc_yaml_path"]], check=True)
        print(f" PVC applied to namespace '{NAMESPACE}'")
    except subprocess.CalledProcessError as e:
        sys.exit(f" Failed to apply PVC YAML: {e}")



def load_profile_list_config(toml_path):
    if not os.path.exists(toml_path):
        raise FileNotFoundError(f"TOML config file not found at: {toml_path}")
    
    config = toml.load(toml_path)
    profile_cfg = config.get("profile_list", {})
    yaml_path = profile_cfg.get("yaml_path")

    if not yaml_path or not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"YAML path not found or invalid in TOML: {yaml_path}")
    
    return yaml_path


def load_profile_config(toml_path):
    config = toml.load(toml_path)
    profile = config.get("profile", {})
    pod_prefix = profile.get("pod_prefix")
    pattern = profile.get("pattern")
    
    if not pod_prefix or not pattern:
        raise ValueError("Both 'pod_prefix' and 'pattern' must be set in [profile] section of the TOML.")
    
    return pod_prefix, pattern

pod_prefix, pattern = load_profile_config(USER_INPUT_PATH)


def create_pod(yaml_path, NAMESPACE):
    print(f"Creating pod from YAML: {yaml_path} in namespace '{NAMESPACE}'")
    result = subprocess.run(
        ["kubectl", "create", "-f", yaml_path, "-n", NAMESPACE],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    if result.returncode != 0:
        if "AlreadyExists" in result.stderr:
            print("Pod already exists. Skipping creation.")
            return
        else:
            print(f"Failed to create pod:\n{result.stderr}")
            raise RuntimeError("kubectl create failed")
    print(result.stdout.strip())



def wait_for_pod_completion(NAMESPACE, pod_prefix, timeout=300):
    print(f"\nWaiting for pod '{pod_prefix}' to reach 'Completed' status...")
    end_time = time.time() + timeout

    while time.time() < end_time:
        result = subprocess.run(
            ["kubectl", "get", "pods", "-n", NAMESPACE, "--no-headers"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )
        if result.returncode != 0:
            print(f"Error getting pods: {result.stderr}")
            time.sleep(5)
            continue

        lines = [line for line in result.stdout.splitlines() if pod_prefix in line]
        if not lines:
            print("Waiting for pod to appear...")
        else:
            line = lines[0]
            pod_name = line.split()[0]
            pod_status = line.split()[2]
            print(f"Current pod '{pod_name}' status: {pod_status}")

            if pod_status == "Completed":
                print(f" Pod '{pod_name}' has completed successfully!")
                return pod_name  # Return the pod name here
            elif pod_status in ("Error", "CrashLoopBackOff"):
                raise RuntimeError(f" Pod '{pod_name}' failed with status: {pod_status}")
        
        time.sleep(5)

    raise TimeoutError(f" Timeout: Pod '{pod_prefix}' did not complete within {timeout} seconds.")


# def fetch_logs_for_pattern(pod_name, NAMESPACE, pattern, toml_path):
#     import re
#     print(f"\nFetching logs for pod '{pod_name}' in namespace '{NAMESPACE}' and filtering for regex pattern '{pattern}'...")

#     result = subprocess.run(
#         ["kubectl", "logs", pod_name, "-n", NAMESPACE],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         universal_newlines=True
#     )

#     if result.returncode != 0:
#         print(f"Failed to get logs:\n{result.stderr}")
#         return

#     regex = re.compile(pattern, re.IGNORECASE)
#     lines = [line for line in result.stdout.splitlines() if regex.search(line)]

#     if not lines:
#         print("No matches found.")
#         return

#     print("\nMatching lines:")
#     for line in lines:
#         print(line)

#     # Extract hash before the first ":" from the first match
#     match_id = lines[0].split(":")[0].strip()

#     # Load existing TOML config
#     config = toml.load(toml_path)
#     if "profile" not in config:
#         config["profile"] = {}
    
#     config["profile"]["selected_model_id"] = match_id

#     # Write updated TOML back
#     with open(toml_path, "w") as f:
#         toml.dump(config, f)

#     print(f"\nUpdated TOML with selected_model_id = {match_id}")


def fetch_profile_pod_logs_and_update_toml(namespace, toml_path):
    try:
        # Load pattern from TOML
        toml_data = toml.load(toml_path)
        pattern = toml_data.get("profile", {}).get("pattern", "")
        if not pattern:
            print(" No pattern found under [profile] in TOML.")
            return

        print(f"\n Using regex pattern from TOML: '{pattern}'")

        # Step 1: Get pods
        result = subprocess.run(
            ["kubectl", "get", "pods", "-n", namespace, "--no-headers"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True
        )

        # Step 2: Find first pod with 'profile' in name
        pods = result.stdout.strip().split('\n')
        profile_pod = None
        for line in pods:
            pod_name = line.split()[0]
            if "profile" in pod_name.lower():
                profile_pod = pod_name
                break

        if not profile_pod:
            print(" No pod found with 'profile' in its name.")
            return

        print(f"\n Found profile pod: {profile_pod}")

        # Step 3: Fetch logs from the pod
        logs_result = subprocess.run(
            ["kubectl", "logs", profile_pod, "-n", namespace],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        if logs_result.returncode != 0:
            print(f" Failed to get logs:\n{logs_result.stderr}")
            return

        # Step 4: Grep pattern from logs
        regex = re.compile(pattern, re.IGNORECASE)
        matches = [line for line in logs_result.stdout.splitlines() if regex.search(line)]

        if not matches:
            print("  No matches found for pattern.")
            return

        print("\n Matching log lines:")
        for line in matches:
            print(line)

        # Step 5: Extract match ID and update TOML
        match_id = matches[0].split(":")[0].strip()
        toml_data["profile"]["selected_model_id"] = match_id

        with open(toml_path, "w") as f:
            toml.dump(toml_data, f)

        print(f"\nUpdated TOML with selected_model_id = {match_id}")

    except subprocess.CalledProcessError as e:
        print(f" Error executing kubectl command:\n{e.stderr}")
    except Exception as e:
        print(f" Unexpected error: {str(e)}")



def delete_temp_pod_from_yaml(yaml_path, NAMESPACE):
    print(f"\nDeleting temporary pod defined in YAML: {yaml_path} from namespace '{NAMESPACE}'...")
    result = subprocess.run(
        ["kubectl", "delete", "-f", yaml_path, "-n", NAMESPACE],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    if result.returncode != 0:
        print(f"Failed to delete pod:\n{result.stderr}")
        return

    print(result.stdout.strip())

    # Optional: Confirm deletion
    time.sleep(3)
    confirm = subprocess.run(
        ["kubectl", "get", "pods", "-n", NAMESPACE],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    print("\nRemaining pods:")
    print(confirm.stdout.strip())

def load_toml_config(toml_path):
    config = toml.load(toml_path)
    download_yaml = config["download"]["download_yaml"]
    image = config["profile"]["image"]
    selected_model_id = config["profile"]["selected_model_id"]
    return download_yaml, image, selected_model_id

def update_download_yaml(yaml_path, image, selected_model_id):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    container = data["spec"]["template"]["spec"]["containers"][0]
    container["image"] = image
    container["args"] = ["download-to-cache", "--profile", selected_model_id]

    with open(yaml_path, 'w') as f:
        yaml.safe_dump(data, f)

    print(f"Updated YAML with image: {image} and profile: {selected_model_id}")

def create_download_job(yaml_path, NAMESPACE):
    print(f"Creating download job from: {yaml_path}")
    result = subprocess.run(
        ["kubectl", "create", "-f", yaml_path, "-n", NAMESPACE],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    if result.returncode != 0:
        if "AlreadyExists" in result.stderr:
            print("Job already exists. Skipping creation.")
        else:
            print(f"Failed to create job:\n{result.stderr}")
            raise RuntimeError("Job creation failed.")
    else:
        print(result.stdout.strip())

# def wait_for_pod_status(namespace, job_name, timeout=300):
#     print(f"Waiting for job pod '{job_name}' to reach 'Running' or 'Completed'...")

#     end_time = time.time() + timeout
#     while time.time() < end_time:
#         result = subprocess.run(
#             ["kubectl", "get", "pods", "-n", namespace, "--no-headers"],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             universal_newlines=True
#         )
#         pods = result.stdout.splitlines()
#         for line in pods:
#             if job_name in line:
#                 status = line.split()[2]
#                 print(f"Pod status: {status}")
#                 if status in ["Running", "Completed"]:
#                     return
#                 elif status in ["Error", "CrashLoopBackOff"]:
#                     raise RuntimeError(f"Pod failed with status: {status}")
#         time.sleep(5)
#     raise TimeoutError(f"Pod did not reach 'Running' or 'Completed' within timeout.")

def run_download_flow(toml_path, NAMESPACE):
    yaml_path, image, selected_model_id = load_toml_config(toml_path)
    update_download_yaml(yaml_path, image, selected_model_id)
    create_download_job(yaml_path, NAMESPACE)

    job_basename = os.path.splitext(os.path.basename(yaml_path))[0].replace("download_", "")
    # wait_for_pod_status(namespace, job_basename)


def load_toml_data(toml_path="user_input.toml"):
    if not os.path.exists(toml_path):
        raise FileNotFoundError(f"TOML file '{toml_path}' not found.")
    return toml.load(toml_path)


def update_runtime_yaml(runtime_yaml_path, image, selected_model_id):
    with open(runtime_yaml_path, "r") as f:
        runtime_yaml = yaml.safe_load(f)

    # Update container image
    runtime_yaml['spec']['containers'][0]['image'] = image

    # Update selected_model_id in env
    for env_var in runtime_yaml['spec']['containers'][0]['env']:
        if env_var['name'] == "NIM_MODEL_PROFILE":
            env_var['value'] = selected_model_id
            break

    with open(runtime_yaml_path, "w") as f:
        yaml.safe_dump(runtime_yaml, f)

    print(" Runtime YAML updated.")


def apply_runtime_yaml(runtime_yaml_path, NAMESPACE):
    print(f" Applying runtime YAML in namespace '{NAMESPACE}'...")
    try:
        subprocess.run(
            ["kubectl", "create", "-f", runtime_yaml_path, "-n", NAMESPACE],
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(" Created runtime successfully.")
    except subprocess.CalledProcessError as e:
        if "AlreadyExists" in e.stderr.decode():
            print("  Runtime already exists. Applying update instead...")
            subprocess.run(
                ["kubectl", "apply", "-f", runtime_yaml_path, "-n", NAMESPACE],
                universal_newlines=True
            )
            print(" Applied update to existing runtime.")
        else:
            raise  # Reraise unexpected errors


def wait_for_clusterservingruntime(NAMESPACE, timeout=180):
    print(f" Waiting for ClusterServingRuntime in namespace '{NAMESPACE}'...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            result = subprocess.check_output(
                ["kubectl", "get", "clusterservingruntime", "-n", NAMESPACE],
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            if "nim" in result or "llama" in result:
                print(" ClusterServingRuntime found:\n", result)
                return
        except subprocess.CalledProcessError:
            pass
        time.sleep(5)
    raise TimeoutError("ClusterServingRuntime not found in time.")


def read_paths_from_toml(toml_file):
    config = toml.load(toml_file)
    runtime_path = config["paths"].get("runtime")
    deploy_path = config["paths"].get("deploy")
    return runtime_path, deploy_path

def update_runtime_in_deploy_yaml(deploy_yaml_path, runtime_name):
    with open(deploy_yaml_path, 'r') as f:
        deploy_config = yaml.safe_load(f)

    deploy_config['spec']['predictor']['model']['runtime'] = runtime_name

    with open(deploy_yaml_path, 'w') as f:
        yaml.safe_dump(deploy_config, f)

    print(f" Updated runtime to '{runtime_name}' in {deploy_yaml_path}")


def create_or_apply_deploy_yaml(deploy_yaml_path, namespace):
    if not deploy_yaml_path or not os.path.isfile(deploy_yaml_path):
        print(f"Deploy YAML path not found or invalid: {deploy_yaml_path}")
        return

    print(f" Creating deploy YAML in namespace '{namespace}' from '{deploy_yaml_path}'...")

    try:
        subprocess.run(
            ["kubectl", "create", "-f", deploy_yaml_path, "-n", namespace],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(" Deploy YAML created successfully.")

    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr.decode() if e.stderr else ""
        print(f" Create failed with error:\n{stderr_output}")

        if "AlreadyExists" in stderr_output:
            print(" InferenceService already exists. Applying update instead...")
            try:
                subprocess.run(
                    ["kubectl", "apply", "-f", deploy_yaml_path, "-n", namespace],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                print(" Deploy YAML applied successfully.")
            except subprocess.CalledProcessError as apply_err:
                print(" Failed to apply updated InferenceService:\n", apply_err.stderr.decode())
                raise
        else:
            raise

    print(f" Checking pod status in namespace '{namespace}'...")
    try:
        subprocess.run(["kubectl", "get", "pods", "-n", namespace], check=True)
    except subprocess.CalledProcessError as e:
        print(" Failed to get pods:\n", e.stderr.decode() if e.stderr else str(e))

# def update_cluster_ip_in_toml(namespace, toml_file_path):
#     try:
#         result = subprocess.run(
#             ["kubectl", "get", "svc", "-n", namespace],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             universal_newlines=True,
#             check=True
#         )
#         lines = result.stdout.strip().split('\n')

#         headers = lines[0].split()
#         name_index = headers.index("NAME")
#         cluster_ip_index = headers.index("CLUSTER-IP")

#         # line ending with 'private'
#         for line in lines[1:]:
#             columns = line.split()
#             name = columns[name_index]
#             if name.endswith("private"):
#                 cluster_ip = columns[cluster_ip_index]
#                 print(f" Found cluster IP for service '{name}': {cluster_ip}")

#                 toml_data = toml.load(toml_file_path)
#                 if "values" not in toml_data:
#                     toml_data["values"] = {}
#                 toml_data["values"]["cluster_ip"] = cluster_ip

#                 with open(toml_file_path, 'w') as f:
#                     toml.dump(toml_data, f)

#                 print(f" Updated 'cluster_ip' in '{toml_file_path}'")
#                 return
#         print(" No service ending with 'private' found.")
#     except subprocess.CalledProcessError as e:
#         print(" Error executing kubectl command:", e.stderr)
#     except Exception as e:
#         print(" Unexpected error:", str(e))


def update_cluster_ip_in_toml(namespace, toml_file_path):
    try:
        result = subprocess.run(
            ["kubectl", "get", "svc", "-n", namespace],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')

        headers = lines[0].split()
        name_index = headers.index("NAME")
        cluster_ip_index = headers.index("CLUSTER-IP")

       
        for line in lines[1:]:
            columns = line.split()
            name = columns[name_index]
            if name.endswith("private") and "8b-bf16-tp1-pp1" in name:
                cluster_ip = columns[cluster_ip_index]
                print(f" Found cluster IP for service '{name}': {cluster_ip}")

                toml_data = toml.load(toml_file_path)
                if "values" not in toml_data:
                    toml_data["values"] = {}
                toml_data["values"]["cluster_ip"] = cluster_ip

                with open(toml_file_path, 'w') as f:
                    toml.dump(toml_data, f)

                print(f" Updated 'cluster_ip' in '{toml_file_path}'")
                return
        print(" No service ending with 'private' and containing '8b-bf16-tp1-pp1' found.")
    except subprocess.CalledProcessError as e:
        print(" Error executing kubectl command:", e.stderr)
    except Exception as e:
        print(" Unexpected error:", str(e))

def create_and_check_pvc(toml_path, namespace):
    config = toml.load(toml_path)
    pvc_yaml_path = config.get("paths", {}).get("workdir_pvc")

    if not pvc_yaml_path:
        raise ValueError("Missing 'workdir_pvc' path under [paths] in the TOML file.")

    print(f" Creating PVC from: {pvc_yaml_path} in namespace '{namespace}'...")
    try:
        subprocess.run(["kubectl", "create", "-f", pvc_yaml_path, "-n", namespace],
                       check=True, stderr=subprocess.PIPE)
        print(" PVC created.")
    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr.decode() if e.stderr else ""
        if "AlreadyExists" in stderr_output:
            print(" PVC already exists. Applying the PVC YAML to ensure it's up to date...")
            try:
                subprocess.run(["kubectl", "apply", "-f", pvc_yaml_path, "-n", namespace], check=True)
                print(" PVC applied.")
            except subprocess.CalledProcessError as apply_err:
                print(" Failed to apply PVC:\n", apply_err)
                raise
        else:
            print(" Failed to create PVC:\n", stderr_output)
            raise

    print(f" Listing PVCs in namespace '{namespace}'...")

    subprocess.run(["kubectl", "get", "pvc", "-n", namespace])


def genai_pod_yaml(toml_path, NAMESPACE):
    config = toml.load(toml_path)
    genai_yaml_path = config.get("paths", {}).get("genai_pod_yaml")

    if not genai_yaml_path:
        raise ValueError("Missing 'genai_pod_yaml' path under [paths] in the TOML file.")

    print(f"Creating {genai_yaml_path} in namespace {NAMESPACE}...")
    try:
        subprocess.run(["kubectl", "create", "-f", genai_yaml_path, "-n", NAMESPACE],
                       check=True, stderr=subprocess.PIPE)
        print("GenAI perf pod created successfully.")
    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr.decode() if e.stderr else ""
        if "AlreadyExists" in stderr_output:
            print("GenAI perf pod already exists. Applying changes...")
            try:
                subprocess.run(["kubectl", "apply", "-f", genai_yaml_path, "-n", NAMESPACE],
                               check=True)
                print("GenAI perf pod applied.")
            except subprocess.CalledProcessError as apply_err:
                print("Failed to apply GenAI perf pod:\n", apply_err)
                raise
        else:
            print("Failed to create GenAI perf pod:\n", stderr_output)
            raise

    print(f"Getting active pods in namespace '{NAMESPACE}'...")
    subprocess.run(["kubectl", "get", "pods", "-n", NAMESPACE])


def exec_into_genai_perf_pod(toml_path,namespace):
        config = toml.load(toml_path)
        cluster_ip = config.get("cluster", {}).get("ip")
        
        get_pods = subprocess.run(
            ["kubectl", "get", "pods", "-n", namespace, "-o", "jsonpath={.items[*].metadata.name}"],
            universal_newlines=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        pods = get_pods.stdout.strip().split()
        target_pod = next((pod for pod in pods if pod.startswith("genai-perf")), None)

        if not target_pod:
            print(f" No pod starting with 'genai-perf' found in namespace '{namespace}'.")
            return

        print(f" Executing into pod: {target_pod}")

        subprocess.run([
            "kubectl", "exec", "-n", namespace, target_pod, "--",
            "bash", "-c", "cd /workdir && df -h"
        ], check=True)

        setup_cmd = (
            "cd /workdir && "
            "mkdir -p llama-3.1-70b-instruct/nim_1.3.2/throughput-bf16-tp1-pp1 && "
            "cd llama-3.1-70b-instruct/nim_1.3.2/throughput-bf16-tp1-pp1 && "
            "pwd"
        )
        subprocess.run([
            "kubectl", "exec", "-n", namespace, target_pod, "--",
            "bash", "-c", setup_cmd
        ], check=True)

        hf_token = config.get("api_keys", {}).get("hugging_face_token")

        if not hf_token:
             print(" 'hugging_face_token' not found in TOML config.")
             return

        login_cmd = f"huggingface-cli login --token {hf_token}"

        subprocess.run([ "kubectl", "exec", "-n", NAMESPACE, target_pod, "--", "bash", "-c", login_cmd], check=True)

        print(" Hugging Face CLI login completed.")

        cluster_ip = config.get("values", {}).get("cluster_ip")

        if not cluster_ip:
         print(" Cluster IP not found in TOML config.")
         return

        url = f"http://{cluster_ip}/v1/models"
        print(f" Checking model serving at: {url}")

        try:
            result = subprocess.run(["curl", "-X", "GET", url], universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
            print(" Response:\n", result.stdout)
        except subprocess.CalledProcessError as e:
            print(" Failed to fetch model status:\n", e.stderr)


import subprocess
import toml

def run_bench_script_from_pod(toml_path, pod_name, namespace):
    config = toml.load(toml_path)

    shell_script = config.get("paths", {}).get("shell_script")
    ip = config.get("values", {}).get("cluster_ip")

    if not shell_script or not ip:
        raise ValueError("Missing 'shell_script' or 'cluster_ip' in TOML file.")

    chmod_cmd = [
        "kubectl", "exec", "-n", namespace, pod_name, "--",
        "chmod", "+x", shell_script
    ]
    subprocess.run(chmod_cmd, check=True)
    print(f" Made script executable: {shell_script}")


    exec_cmd = [
    "kubectl", "exec", "-n", NAMESPACE, pod_name, "--",
    "bash", shell_script,
    "--model", "meta/llama-3.1-8b-instruct",
    "--measurement-interval", "60000",
    "--tokenizer", "meta-llama/Llama-3.1-8B-Instruct",
    "--url", f"http://{ip}",
    "--export-file-name", "test3-madhu",
    "--concurrency-values", "1,2,4",
    "--use-cases", "Search"
]



    print(f" Running benchmark script inside pod '{pod_name}'...")
    subprocess.run(exec_cmd, check=True)
    print(" Benchmark script executed successfully.")



def main():
    print(" TOML Test Scheduler Started!\n")
    config = toml.load(USER_INPUT_PATH)
    test_config = load_input_config(USER_INPUT_PATH)

    print(" Creating test_config.toml file...")
    create_tests(test_config)

    print(" Executing test scheduler...")
    run_scheduler()

    api_keys = config.get("api_keys", {})
   
    print(" Exporting API keys as environment variables...")
    ngc_api_key, ngc_token, hf_token = export_env_vars(api_keys)

    if not ngc_api_key:
        raise ValueError(" Missing 'ngc_api_key' in [api_keys]")
    if not hf_token:
        raise ValueError(" Missing 'hugging_face_token' in [api_keys]")


    print(f" Checking if namespace '{NAMESPACE}' exists...")
    ns_check = subprocess.run( ["kubectl", "get", "ns", NAMESPACE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


    if ns_check.returncode != 0:
     print(f" Creating Kubernetes namespace '{NAMESPACE}'...")
     subprocess.run(["kubectl", "create", "ns", NAMESPACE], check=True)
    else:
     print(f" Namespace '{NAMESPACE}' already exists. Skipping creation.")


    print(f" Checking if Docker registry secret 'ngc-secret' exists in namespace '{NAMESPACE}'...")
    secret_check = subprocess.run( ["kubectl", "get", "secret", "ngc-secret", "-n", NAMESPACE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


    print(f"Checking if Docker registry secret 'ngc-secret' exists in namespace '{NAMESPACE}'...")
    secret_check = subprocess.run(
    ["kubectl", "get", "secret", "ngc-secret", "-n", NAMESPACE],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

    if secret_check.returncode == 0:
      print("Deleting existing 'ngc-secret' to update it with new key...")
      subprocess.run(["kubectl", "delete", "secret", "ngc-secret", "-n", NAMESPACE], check=True)

    print("Creating Docker registry secret 'ngc-secret'...")
    subprocess.run([
    "kubectl", "create", "secret", "docker-registry", "ngc-secret",
    "-n", NAMESPACE,
    "--docker-server=nvcr.io",
    "--docker-username", "oauthtoken",
    "--docker-password", ngc_api_key
], check=True)

     
    print(f" Checking if secret 'nvidia-nim-secrets' exists in namespace '{NAMESPACE}'...")
    nim_secret_check = subprocess.run(
        ["kubectl", "get", "secret", "nvidia-nim-secrets", "-n", NAMESPACE],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    if nim_secret_check.returncode == 0:
        print(" Secret 'nvidia-nim-secrets' already exists. Deleting it to recreate...")
        subprocess.run(["kubectl", "delete", "secret", "nvidia-nim-secrets", "-n", NAMESPACE], check=True)
    else:
        print(" 'nvidia-nim-secrets' does not exist. Proceeding to create it...")

    print(" Creating Kubernetes secret 'nvidia-nim-secrets'...")
    subprocess.run([
        "kubectl", "create", "secret", "generic", "nvidia-nim-secrets",
        "-n", NAMESPACE,
        f"--from-literal=token={hf_token}",
        f"--from-literal=api-key={ngc_api_key}"
    ], check=True)
   
    nim_secrets_yaml_path = Path("/home/pcadmin/cloudai/nim-deploy/kserve/scripts/nvidia-nim-secrets.yaml")

    if not nim_secrets_yaml_path.is_file():
     raise FileNotFoundError(f"YAML template file not found or not a file: {nim_secrets_yaml_path}")

    # Encode secrets
    print(" Encoding HF_TOKEN and NGC_API_KEY as base64...")
    hf_token_b64 = base64.b64encode(hf_token.encode()).decode()
    ngc_api_key_b64 = base64.b64encode(ngc_api_key.encode()).decode()

    # Read and substitute YAML template
    print(f" Reading YAML secret template from: {nim_secrets_yaml_path}")
    with open(nim_secrets_yaml_path, "r") as f:
        yaml_content = f.read()

    print(" Replacing placeholders with encoded secrets...")
    yaml_content = yaml_content.replace("${HF_TOKEN}", hf_token_b64)
    yaml_content = yaml_content.replace("${NGC_API_KEY}", ngc_api_key_b64)

    # Apply updated secret YAML to Kubernetes
    print(f" Applying updated secret YAML to namespace '{NAMESPACE}'...")
    subprocess.run(
        ["kubectl", "apply", "-n", NAMESPACE, "-f", "-"], input=yaml_content, universal_newlines=True, check=True)
    print(f" Successfully applied secret to Kubernetes namespace '{NAMESPACE}'\n")

    result_label = subprocess.run(
    ["kubectl", "get", "ns", NAMESPACE, "--show-labels"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    print(result_label.stdout)

    print(f" Labeling namespace '{NAMESPACE}'...")
    subprocess.run(["kubectl", "label", "ns", NAMESPACE, "hpe-ezua/ezmodels=true", "--overwrite"], check=True)

    pvc_config = config.get('pvc_details', {})
    update_pvc_yaml(pvc_config)

    print(" Starting pod creation and monitoring process...")
    yaml_path = load_profile_list_config(USER_INPUT_PATH)

    create_pod(yaml_path, NAMESPACE)

    pod_name_prefix = "list-profiles-llama-31-8b"
    
    completed_pod = wait_for_pod_completion(NAMESPACE, pod_prefix)
    print("\n Script completed.")
    # fetch_logs_for_pattern(completed_pod, NAMESPACE, pattern,USER_INPUT_PATH)
    fetch_profile_pod_logs_and_update_toml(NAMESPACE, USER_INPUT_PATH)

    delete_temp_pod_from_yaml(yaml_path, NAMESPACE)


    # wait_for_pod_status(NAMESPACE, job_name, timeout=300)
    run_download_flow(USER_INPUT_PATH, NAMESPACE)

    data = load_toml_data(USER_INPUT_PATH)

    runtime_yaml = data["paths"]["runtime"]
    image = data["profile"]["image"]
    selected_model_id = data["profile"]["selected_model_id"]

    update_runtime_yaml(runtime_yaml, image, selected_model_id)
    apply_runtime_yaml(runtime_yaml, NAMESPACE)
    wait_for_clusterservingruntime(NAMESPACE)   


    runtime_yaml, deploy_yaml = read_paths_from_toml(USER_INPUT_PATH)

    runtime_name = "madhu-31-bf16-tp1-pp1-tpt"

    update_runtime_in_deploy_yaml(deploy_yaml, runtime_name)
   

    # deploy_yaml_path = config["paths"].get("deploy")

    # if not deploy_yaml_path or not os.path.isfile(deploy_yaml_path):
    #     print(f" Deploy YAML path not found: {deploy_yaml_path}")
    #     return
    # else:
    #  create_or_apply_deploy_yaml(deploy_yaml_path, NAMESPACE)

    # config = toml.load("user_input.toml")

# Read required values
    deploy_yaml_path = config["paths"].get("deploy")
    

    create_or_apply_deploy_yaml(deploy_yaml_path, NAMESPACE)
    get_pods = subprocess.run(
            ["kubectl", "get", "pods", "-n",NAMESPACE, "-o", "jsonpath={.items[*].metadata.name}"],
            universal_newlines=True,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,)
    pods = get_pods.stdout.strip().split()
    target_pod = next((pod for pod in pods if pod.startswith("genai-perf")), None)
    update_cluster_ip_in_toml(NAMESPACE, USER_INPUT_PATH)  

    create_and_check_pvc(USER_INPUT_PATH, NAMESPACE)  

    genai_pod_yaml(USER_INPUT_PATH, NAMESPACE)

    exec_into_genai_perf_pod(USER_INPUT_PATH,NAMESPACE)

    # run_bench_script(USER_INPUT_PATH)
        

    run_bench_script_from_pod(USER_INPUT_PATH, target_pod, NAMESPACE)

if __name__ == "__main__":
    main()
