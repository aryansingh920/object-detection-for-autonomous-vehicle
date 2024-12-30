#!/bin/zsh

# Enable nullglob to prevent errors when no files match the glob pattern
setopt nullglob

# Optional: Enable debugging by uncommenting the next line
# set -x

# Use the current directory if no argument is provided
DIR=${1:-.}

# Threshold for summarizing file extensions
THRESHOLD=10

# Function to recursively print the directory structure
print_tree() {
    local dir=$1
    local indent=$2

    # Declare an associative array to count file extensions for this directory
    typeset -A extension_count
    # Initialize an array to hold files to be listed
    typeset -a files_to_list

    # First pass: Count the number of files per extension
    for item in "$dir"/* "$dir"/.*; do
        # Skip '.' and '..' entries
        basename_item=$(basename "$item")
        if [[ "$basename_item" == "." || "$basename_item" == ".." ]]; then
            continue
        fi

        if [ -f "$item" ]; then
            # Extract extension and normalize to lowercase
            local ext="${item##*.}"
            ext="${ext:l}"  # Convert to lowercase

            # Handle files without an extension or hidden files without an extension
            if [[ "$item" == .* && "$ext" == "$item" ]]; then
                ext="no_extension"
            elif [ "$ext" = "$item" ]; then
                ext="no_extension"
            fi

            # Increment the count for this extension
            ((extension_count["$ext"]++))
        fi
    done

    # Second pass: Decide whether to list files or show counts
    for item in "$dir"/* "$dir"/.*; do
        # Skip '.' and '..' entries
        basename_item=$(basename "$item")
        if [[ "$basename_item" == "." || "$basename_item" == ".." ]]; then
            continue
        fi

        if [ -d "$item" ]; then
            # Print the directory name
            echo "${indent}|-- $(basename "$item")/"
            # Recursively call the function for the subdirectory
            print_tree "$item" "${indent}   "
        elif [ -f "$item" ]; then
            # Extract and normalize extension
            local ext="${item##*.}"
            ext="${ext:l}"  # Convert to lowercase

            # Handle files without an extension or hidden files without an extension
            if [[ "$item" == .* && "$ext" == "$item" ]]; then
                ext="no_extension"
            elif [ "$ext" = "$item" ]; then
                ext="no_extension"
            fi

            # Only add to list if the count is <= threshold
            if [ "${extension_count[$ext]}" -le "$THRESHOLD" ]; then
                # Collect files to list later to avoid interleaving with directories
                files_to_list+=("$(basename "$item")")
            fi
        fi
    done

    # Print the files that are below the threshold
    for file in "${files_to_list[@]}"; do
        echo "${indent}|-- $file"
    done

    # After listing all items, print summaries for extensions with more than the threshold
    for ext in ${(k)extension_count}; do
        if [ "${extension_count[$ext]}" -gt "$THRESHOLD" ]; then
            if [ "$ext" = "no_extension" ]; then
                echo "${indent}|-- More than $THRESHOLD files with no extension: ${extension_count[$ext]} files"
            else
                echo "${indent}|-- More than $THRESHOLD files with the extension .$ext: ${extension_count[$ext]} files"
            fi
        fi
    done
}

# Print the root directory
echo "$(basename "$DIR")/"
print_tree "$DIR" "   "
