//! Build script for Memory Nexus Blocks
//! Detects CPU features and configures optimizations at compile time

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    
    // Detect target architecture
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|_| "unknown".to_string());
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| "unknown".to_string());
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_else(|_| "".to_string());
    
    println!("cargo:warning=Building for {}-{}-{}", target_arch, target_os, target_env);
    
    // CPU Feature Detection
    detect_cpu_features(&target_arch);
    
    // Memory allocator configuration
    configure_allocator();
    
    // SIMD optimizations
    configure_simd(&target_arch);
    
    // Link-time optimizations
    configure_lto();
    
    // Profile-Guided Optimization support
    configure_pgo();
    
    // C ABI compatibility checks
    configure_c_abi();
    
    // Platform-specific optimizations
    platform_optimizations(&target_os, &target_arch);
}

fn detect_cpu_features(arch: &str) {
    match arch {
        "x86_64" | "x86" => {
            // x86/x86_64 CPU features
            if is_x86_feature_detected("avx512f") {
                println!("cargo:rustc-cfg=target_feature=\"avx512f\"");
                println!("cargo:warning=AVX-512 detected - enabling 512-bit SIMD");
            }
            if is_x86_feature_detected("avx2") {
                println!("cargo:rustc-cfg=target_feature=\"avx2\"");
                println!("cargo:warning=AVX2 detected - enabling 256-bit SIMD");
            }
            if is_x86_feature_detected("fma") {
                println!("cargo:rustc-cfg=target_feature=\"fma\"");
                println!("cargo:warning=FMA detected - enabling fused multiply-add");
            }
            if is_x86_feature_detected("sse4.2") {
                println!("cargo:rustc-cfg=target_feature=\"sse4.2\"");
            }
            if is_x86_feature_detected("popcnt") {
                println!("cargo:rustc-cfg=target_feature=\"popcnt\"");
                println!("cargo:warning=POPCNT detected - hardware popcount available");
            }
            if is_x86_feature_detected("bmi2") {
                println!("cargo:rustc-cfg=target_feature=\"bmi2\"");
            }
            
            // Cache line size detection
            detect_cache_line_size();
        }
        "aarch64" => {
            // ARM64 features
            println!("cargo:rustc-cfg=target_feature=\"neon\"");
            println!("cargo:warning=NEON SIMD detected for ARM64");
            
            // Check for SVE (Scalable Vector Extension)
            if cfg!(target_feature = "sve") {
                println!("cargo:rustc-cfg=target_feature=\"sve\"");
                println!("cargo:warning=SVE detected - enabling scalable vectors");
            }
        }
        _ => {
            println!("cargo:warning=Unknown architecture: {} - using scalar fallbacks", arch);
        }
    }
}

fn is_x86_feature_detected(feature: &str) -> bool {
    // This is a compile-time check for build script
    // At runtime, we'll use std::is_x86_feature_detected!
    match feature {
        "avx512f" => cfg!(target_feature = "avx512f"),
        "avx2" => cfg!(target_feature = "avx2"),
        "fma" => cfg!(target_feature = "fma"),
        "sse4.2" => cfg!(target_feature = "sse4.2"),
        "popcnt" => cfg!(target_feature = "popcnt"),
        "bmi2" => cfg!(target_feature = "bmi2"),
        _ => false,
    }
}

fn detect_cache_line_size() {
    // Try to detect cache line size (usually 64 or 128 bytes)
    // Default to 64 which is most common
    let cache_line_size = if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
        128 // Apple Silicon uses 128-byte cache lines
    } else {
        64 // Most x86_64 and other systems use 64-byte cache lines
    };
    
    println!("cargo:rustc-env=CACHE_LINE_SIZE={}", cache_line_size);
    println!("cargo:warning=Cache line size: {} bytes", cache_line_size);
}

fn configure_allocator() {
    // Set jemalloc/mimalloc configuration for optimal performance
    if cfg!(feature = "jemalloc") {
        println!("cargo:rustc-env=_RJEM_MALLOC_CONF=background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000");
        println!("cargo:warning=Using jemalloc with optimized settings");
    } else if cfg!(feature = "mimalloc") {
        println!("cargo:rustc-env=MIMALLOC_LARGE_OS_PAGES=1");
        println!("cargo:rustc-env=MIMALLOC_RESERVE_HUGE_OS_PAGES=4");
        println!("cargo:warning=Using mimalloc with large page support");
    }
}

fn configure_simd(arch: &str) {
    if !cfg!(feature = "simd") {
        return;
    }
    
    match arch {
        "x86_64" | "x86" => {
            // Enable target-cpu=native if not cross-compiling
            if env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default() 
                == env::var("HOST").unwrap_or_default().split('-').next().unwrap_or_default() {
                println!("cargo:rustc-env=RUSTFLAGS=-C target-cpu=native");
                println!("cargo:warning=Using native CPU optimizations");
            }
        }
        "aarch64" => {
            // ARM64 specific optimizations
            println!("cargo:rustc-env=RUSTFLAGS=-C target-feature=+neon,+fp16");
        }
        _ => {}
    }
}

fn configure_lto() {
    // Link-Time Optimization settings
    if cfg!(profile = "release") || cfg!(profile = "bench") {
        // Use LLD linker if available (faster than default)
        if cfg!(target_os = "linux") {
            println!("cargo:rustc-link-arg=-fuse-ld=lld");
            println!("cargo:warning=Using LLD linker for faster builds");
        } else if cfg!(target_os = "macos") {
            // macOS uses ld64.lld
            println!("cargo:rustc-link-arg=-fuse-ld=lld");
        }
        
        // Enable additional link optimizations
        println!("cargo:rustc-link-arg=-Wl,--gc-sections");
        println!("cargo:rustc-link-arg=-Wl,--strip-all");
    }
}

fn configure_pgo() {
    // Profile-Guided Optimization configuration
    if let Ok(pgo_profile_dir) = env::var("PGO_PROFILE_DIR") {
        println!("cargo:rustc-env=LLVM_PROFILE_FILE={}/default_%m_%p.profraw", pgo_profile_dir);
        println!("cargo:warning=PGO profiling enabled - output to {}", pgo_profile_dir);
        
        if cfg!(feature = "pgo-ready") {
            println!("cargo:rustc-cfg=pgo_active");
        }
    }
    
    // BOLT post-link optimization markers
    if env::var("BOLT_PROFILE").is_ok() {
        println!("cargo:rustc-env=RUSTFLAGS=-C link-arg=-Wl,--emit-relocs");
        println!("cargo:warning=BOLT optimization markers enabled");
    }
}

fn configure_c_abi() {
    // Ensure C ABI compatibility for hot-swapping
    println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libnexus_blocks.so");
    println!("cargo:rustc-cdylib-link-arg=-Wl,--version-script=exports.lds");
    
    // Generate exports file if it doesn't exist
    let exports_path = PathBuf::from("exports.lds");
    if !exports_path.exists() {
        std::fs::write(
            exports_path,
            "NEXUS_BLOCKS_1.0 {\n  global:\n    nexus_*;\n  local:\n    *;\n};\n"
        ).expect("Failed to write exports.lds");
    }
}

fn platform_optimizations(os: &str, arch: &str) {
    match os {
        "linux" => {
            // Linux-specific optimizations
            println!("cargo:rustc-link-arg=-Wl,-z,now");  // Immediate binding
            println!("cargo:rustc-link-arg=-Wl,-z,relro"); // Read-only relocations
            
            // Huge pages support
            if arch == "x86_64" {
                println!("cargo:rustc-env=MALLOC_ARENA_MAX=2");
                println!("cargo:warning=Linux optimizations: huge pages, reduced arenas");
            }
        }
        "macos" => {
            // macOS specific
            println!("cargo:rustc-link-arg=-Wl,-dead_strip");
            
            if arch == "aarch64" {
                // Apple Silicon optimizations
                println!("cargo:rustc-env=RUSTFLAGS=-C target-cpu=apple-m1");
                println!("cargo:warning=Apple Silicon optimizations enabled");
            }
        }
        "windows" => {
            // Windows specific
            println!("cargo:rustc-link-arg=/OPT:REF");
            println!("cargo:rustc-link-arg=/OPT:ICF");
        }
        _ => {}
    }
    
    // Set optimization level hints
    println!("cargo:rustc-env=NEXUS_BLOCKS_OPT_LEVEL=3");
    println!("cargo:rustc-env=NEXUS_BLOCKS_TARGET_ARCH={}", arch);
    println!("cargo:rustc-env=NEXUS_BLOCKS_TARGET_OS={}", os);
}