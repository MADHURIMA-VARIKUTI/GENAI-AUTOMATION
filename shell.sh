
#!/usr/bin/bash

# Default values
DEFAULT_URL="http://127.0.0.1"
DEFAULT_MEASUREMENT_INTERVAL="300000" # in milliseconds
DEFAULT_MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
DEFAULT_TOKENIZER="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
DEFAULT_USE_CASES=("Search" "Summarization" "Translation")
DEFAULT_CONCURRENCY_VALUES=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)

# Use case definitions
declare -A useCases=(
    ["dell"]="128/128"
    ["Search"]="200/1000"
    ["Summarization"]="7000/1000"
    ["Translation"]="200/200"
    ["LargeCL-130500"]="130500/200"
    ["LargeCL-100000"]="100000/200"
    ["LargeCL-80000"]="80000/200"
    ["LargeCL-60000"]="80000/200"
    ["LargeCL-40000"]="40000/200"
    ["LargeCL-20000"]="20000/200"
    ["LargeCL-10000"]="10000/200"
    ["LargeCL-8192"]="8192/200"
)

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --url) URL="$2"; shift ;;
        --measurement-interval) MEASUREMENT_INTERVAL="$2"; shift ;;
        --use-cases) IFS=',' read -r -a USE_CASES_LIST <<< "$2"; shift ;;
        --concurrency-values) IFS=',' read -r -a CONCURRENCY_VALUES <<< "$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --tokenizer) TOKENIZER="$2"; shift ;;
        --export-file-name) EXPORT_FILE_NAME="$2"; shift ;;
        --get-results) GET_RESULTS=true ;;
        --artifacts-dir) ARTIFACTS_DIR="$2"; shift ;;
        --service-type) SERVICE_TYPE="$2"; shift ;;
        --endpoint-type) ENDPOINT_TYPE="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Check if --artifacts-dir is mandatory with --get-results
if [[ "$GET_RESULTS" == true && -z "$ARTIFACTS_DIR" ]]; then
    echo "Error: --artifacts-dir is required when --get-results is specified."
    exit 1
fi

# Apply defaults if not provided
URL="${URL:-$DEFAULT_URL}"
MEASUREMENT_INTERVAL="${MEASUREMENT_INTERVAL:-$DEFAULT_MEASUREMENT_INTERVAL}"
MODEL="${MODEL:-$DEFAULT_MODEL}"
TOKENIZER="${TOKENIZER:-$DEFAULT_TOKENIZER}"
USE_CASES_LIST=("${USE_CASES_LIST[@]:-${DEFAULT_USE_CASES[@]}}")
CONCURRENCY_VALUES=("${CONCURRENCY_VALUES[@]:-${DEFAULT_CONCURRENCY_VALUES[@]}}")
GET_RESULTS="${GET_RESULTS:-false}"
SERVICE_TYPE="${SERVICE_TYPE:-openai}"
ENDPOINT_TYPE="${ENDPOINT_TYPE:-chat}"
FILE_TAG="genai_perf.json"

# Validate required arguments
if [[ -z "$EXPORT_FILE_NAME" ]]; then
    echo "Error: --export-file-name argument is required."
    exit 1
fi

# Validate use cases
for use_case in "${USE_CASES_LIST[@]}"; do
    if [[ -z "${useCases[$use_case]}" ]]; then
        echo "Warning: Use case '$use_case' is not defined."
    fi
done

# Function to print parameters in a box
printParametersBox() {
    local description="$1"
    local concurrency="$2"
    local inputLength="$3"
    local outputLength="$4"
    local export_file="$5"

    echo "========================================"
    echo "Benchmark Parameters:"
    echo "----------------------------------------"
    echo "Use Case:          $description"
    echo "Concurrency:       $concurrency"
    echo "Input Tokens:      $inputLength"
    echo "Output Tokens:     $outputLength"
    echo "Model:             $MODEL"
    echo "Tokenizer:         $TOKENIZER"
    echo "URL:               $URL"
    echo "Measurement Intvl: $MEASUREMENT_INTERVAL ms"
    echo "Export File:       $export_file"
    echo "========================================"
}


read_json() {
    local file_path="$1"
    shift
    local fields=("$@")

    if [[ ! -f "$file_path" ]]; then
        echo "Error: File '$file_path' not found." >&2
        return 1
    fi

    local results=()
    for field in "${fields[@]}"; do
        local value
        value=$(jq "${field}"  "$file_path" 2>/dev/null)
        if [[ $? -ne 0 || "$value" == "N/A" ]]; then
            echo "Error: Failed to parse field '$field' from '$file_path'." >&2
            return 1
        fi
        results+=("$value")
    done

    # Join results with commas and output
    IFS=','; echo "${results[*]}"
}


getResults() {
    local artifacts_dir="$1"
    local export_file_pattern="$2"
    local description="$3"
    local lengths="${useCases[$description]}"
    IFS='/' read -r inputLength outputLength <<< "$lengths"
    shift 3
    local concurrency_values=("$@")
    local results_file="${export_file_pattern}-results.csv"
    local model="${MODEL//\//_}"

    echo -n "Use Case, Concurrency, Input Tokens, Output Tokens,"
    echo -n "TTFT Average, TTFT Min, TTFT Max, TTFT 90th Percentile,"
    echo -n "ITL Average, ITL Min, ITL Max, ITL 90th Percentile,"
    echo -n "Request Latency Avg, Request Latency Min, Average Latency Max, Request Latency 90th Percentile,"
    echo  "Output Token Throughput, Request Throughput, Request Count for BM"
    for concurrency in "${CONCURRENCY_VALUES[@]}"; do
        local file="${artifacts_dir}/${model}-${SERVICE_TYPE}-${ENDPOINT_TYPE}-concurrency${concurrency}/${export_file_pattern}_${use_case}_${concurrency}_${inputLength}_${outputLength}_${FILE_TAG}"
        # echo $file
        local values
        values=$(read_json "$file" ".time_to_first_token.avg" ".time_to_first_token.min" ".time_to_first_token.max" ".time_to_first_token.p90" \
            ".inter_token_latency.avg" ".inter_token_latency.min" ".inter_token_latency.max" ".inter_token_latency.p90" \
            ".request_latency.avg" ".request_latency.min" ".request_latency.max" ".request_latency.p90" \
            ".output_token_throughput.avg" ".request_throughput.avg" ".request_count.avg")
        echo -n "${description}," | tee -a "$results_file"
        echo -n "${concurrency}," | tee -a "$results_file"
        echo -n "${inputLength}," | tee -a "$results_file"
        echo -n "${outputLength}," | tee -a "$results_file"
        echo "$values" | tee -a "$results_file"
     done
}

# Benchmark function
runBenchmark() {
    local description="$1"
    local lengths="${useCases[$description]}"
    IFS='/' read -r inputLength outputLength <<< "$lengths"

    for concurrency in "${CONCURRENCY_VALUES[@]}"; do
        local export_file="${EXPORT_FILE_NAME}_${description}_${concurrency}_${inputLength}_${outputLength}.json"
        printParametersBox "$description" "$concurrency" "$inputLength" "$outputLength" "$export_file"
        genai-perf profile \
            -m "$MODEL" \
            --endpoint-type chat \
            --streaming \
            --num-prompts "$concurrency" \
            --random-seed 1234 \
            -u "$URL" \
            --synthetic-input-tokens-mean "$inputLength" \
            --synthetic-input-tokens-stddev 0 \
            --concurrency "$concurrency" \
            --output-tokens-mean "$outputLength" \
            --output-tokens-stddev 0 \
            --extra-inputs max_tokens:"$outputLength" \
            --extra-inputs min_tokens:"$outputLength" \
            --extra-inputs ignore_eos:true \
            --tokenizer "$TOKENIZER" \
            --measurement-interval "$MEASUREMENT_INTERVAL" \
            --profile-export-file "$export_file" \
            -v \
            -- \
            -v
    done
}

# Check if --get-results is enabled
if [[ "$GET_RESULTS" == true ]]; then
    # Call getResults function with appropriate arguments
    for use_case in "${USE_CASES_LIST[@]}"; do
        if [[ -n "${useCases[$use_case]}" ]]; then
            getResults "$ARTIFACTS_DIR" "$EXPORT_FILE_NAME" "$use_case" "${CONCURRENCY_VALUES[@]}"
        fi
    done
else
    # Run benchmarks for specified use cases
    for use_case in "${USE_CASES_LIST[@]}"; do
        if [[ -n "${useCases[$use_case]}" ]]; then
            runBenchmark "$use_case"
        fi
    done
fi
