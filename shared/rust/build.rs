// ============================================================
// Build script for generating Protobuf code
// ============================================================

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get the manifest directory (the root of the crate)
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);

    // Path to the proto files (relative to the manifest dir)
    let proto_dir = manifest_dir.join("../proto");
    // Use OUT_DIR for generated code (Cargo's build directory)
    let out_dir = env::var("OUT_DIR")?;
    let out_path = PathBuf::from(&out_dir);

    // Create the output directory if it doesn't exist
    if !out_path.exists() {
        fs::create_dir_all(&out_path)?;
    }

    // List all .proto files in the proto directory
    let proto_files: Vec<PathBuf> = fs::read_dir(&proto_dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("proto") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if proto_files.is_empty() {
        eprintln!("⚠️ No .proto files found in {}", proto_dir.display());
        return Ok(());
    }

    println!("cargo:warning=Found {} proto files", proto_files.len());
    println!("cargo:warning=OUT_DIR: {}", out_dir);

    // Compile the protos using tonic-build, output to OUT_DIR
    tonic_build::configure()
        .out_dir(&out_dir)
        .compile(&proto_files, &[proto_dir])?;

    // List generated files for debugging
    println!("cargo:warning=Generated files in OUT_DIR:");
    if out_path.exists() {
        for entry in fs::read_dir(&out_path)? {
            let entry = entry?;
            println!("cargo:warning= - {}", entry.file_name().to_string_lossy());
        }
    } else {
        println!("cargo:warning=OUT_DIR does not exist!");
    }

    // Tell Cargo to rerun this script if any proto file changes
    for file in &proto_files {
        println!("cargo:rerun-if-changed={}", file.display());
    }

    Ok(())
}
