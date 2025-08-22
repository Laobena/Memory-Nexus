use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    
    // Enable CPU-specific optimizations
    detect_cpu_features();
    
    // Set optimization flags
    set_optimization_flags();
    
    // Configure allocator
    configure_allocator();
    
    // Set link-time optimizations
    configure_linking();
    
    // Platform-specific optimizations
    configure_platform();
}

/// Comprehensive CPU feature detection
fn detect_cpu_features() {
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    
    if target_arch == "x86_64" {
        // Check for AVX2 support
        if is_x86_feature_detected("avx2") {
            println!("cargo:rustc-cfg=has_avx2");
            println!("cargo:rustc-cfg=has_simd");
            println!("cargo:warning=✅ AVX2 support detected - enabling optimizations");
        }
        
        // Check for AVX-512 support
        if is_x86_feature_detected("avx512f") {
            println!("cargo:rustc-cfg=has_avx512");
            println!("cargo:rustc-cfg=has_avx512f");
            println!("cargo:warning=✅ AVX-512 support detected - enabling advanced optimizations");
        }
        
        // Check for FMA support
        if is_x86_feature_detected("fma") {
            println!("cargo:rustc-cfg=has_fma");
            println!("cargo:warning=✅ FMA support detected - enabling fused multiply-add");
        }
        
        // Check for SSE4.2 support
        if is_x86_feature_detected("sse4.2") || is_x86_feature_detected("sse4_2") {
            println!("cargo:rustc-cfg=has_sse");
            println!("cargo:rustc-cfg=has_sse42");
            if !is_x86_feature_detected("avx2") {
                println!("cargo:rustc-cfg=has_simd");
            }
        }
        
        // Check for POPCNT support (for binary embeddings)
        if is_x86_feature_detected("popcnt") {
            println!("cargo:rustc-cfg=has_popcnt");
            println!("cargo:warning=✅ POPCNT support detected - enabling fast Hamming distance");
        }
        
        // Check for BMI2 support
        if is_x86_feature_detected("bmi2") {
            println!("cargo:rustc-cfg=has_bmi2");
        }
        
        // Enable native CPU optimizations if not cross-compiling
        if !is_cross_compiling() {
            println!("cargo:rustc-env=RUSTFLAGS=-C target-cpu=native");
            println!("cargo:warning=🚀 Native CPU optimizations enabled");
        }
    } else if target_arch == "aarch64" {
        println!("cargo:rustc-cfg=has_neon");
        println!("cargo:rustc-cfg=has_simd");
        println!("cargo:warning=✅ NEON support detected for ARM64");
        
        if !is_cross_compiling() {
            println!("cargo:rustc-env=RUSTFLAGS=-C target-cpu=native");
        }
    } else if target_arch == "wasm32" {
        println!("cargo:rustc-cfg=has_wasm_simd");
        println!("cargo:warning=✅ WebAssembly SIMD support enabled");
    }
    
    // Detect CPU core count for parallel processing
    if let Ok(num_cpus) = std::thread::available_parallelism() {
        println!("cargo:rustc-env=CPU_CORES={}", num_cpus.get());
        println!("cargo:warning=📊 Detected {} CPU cores", num_cpus.get());
    }
    
    // Detect cache line size for alignment optimizations
    detect_cache_line_size();
}

/// Platform-optimized feature detection for x86
fn is_x86_feature_detected(feature: &str) -> bool {
    // First try environment variable (for CI/CD)
    if let Ok(features) = env::var("CARGO_CFG_TARGET_FEATURE") {
        if features.contains(feature) {
            return true;
        }
    }
    
    // Then try compile-time detection
    #[cfg(target_arch = "x86_64")]
    {
        match feature {
            "avx2" => is_x86_feature_detected!("avx2"),
            "avx512f" => is_x86_feature_detected!("avx512f"),
            "sse4.2" | "sse4_2" => is_x86_feature_detected!("sse4.2"),
            "fma" => is_x86_feature_detected!("fma"),
            "popcnt" => is_x86_feature_detected!("popcnt"),
            "bmi2" => is_x86_feature_detected!("bmi2"),
            _ => false,
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        // Fallback: try to detect via /proc/cpuinfo on Linux
        if cfg!(target_os = "linux") {
            Command::new("sh")
                .arg("-c")
                .arg(format!("grep -q {} /proc/cpuinfo", feature))
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        } else {
            false
        }
    }
}

/// Set optimization flags based on build profile
fn set_optimization_flags() {
    let profile = env::var("PROFILE").unwrap_or_default();
    
    match profile.as_str() {
        "release" => {
            // Enable aggressive optimizations for release builds
            println!("cargo:rustc-link-arg=-flto=fat");
            println!("cargo:rustc-link-arg=-fuse-ld=lld");
            
            // Enable CPU-specific optimizations if not cross-compiling
            if !is_cross_compiling() {
                println!("cargo:rustc-flags=-C target-cpu=native");
                println!("cargo:rustc-flags=-C opt-level=3");
                println!("cargo:rustc-flags=-C codegen-units=1");
            }
            
            // Enable profile-guided optimization if available
            if let Ok(pgo_dir) = env::var("PGO_PROFILE_DIR") {
                println!("cargo:rustc-cfg=pgo_enabled");
                println!("cargo:rustc-flags=-C profile-use={}", pgo_dir);
                println!("cargo:warning=📈 PGO enabled with profile: {}", pgo_dir);
            }
            
            // Enable BOLT optimization if available
            if env::var("BOLT_PROFILE").is_ok() {
                println!("cargo:rustc-cfg=bolt_enabled");
                println!("cargo:warning=⚡ BOLT optimization enabled");
            }
            
            println!("cargo:warning=🎯 Release build optimizations enabled");
        }
        "dev-fast" => {
            // Fast development builds with minimal optimization
            println!("cargo:rustc-flags=-C opt-level=1");
            println!("cargo:rustc-flags=-C codegen-units=256");
            println!("cargo:rustc-flags=-C debuginfo=0");
            println!("cargo:warning=⚡ Fast development build enabled");
        }
        "bench" => {
            // Benchmark-specific optimizations
            println!("cargo:rustc-flags=-C opt-level=3");
            println!("cargo:rustc-flags=-C lto=fat");
            println!("cargo:rustc-flags=-C codegen-units=1");
            if !is_cross_compiling() {
                println!("cargo:rustc-flags=-C target-cpu=native");
            }
            println!("cargo:warning=📊 Benchmark optimizations enabled");
        }
        _ => {
            // Default development build
            println!("cargo:warning=🔧 Development build");
        }
    }
}

/// Configure memory allocator based on features
fn configure_allocator() {
    let use_mimalloc = env::var("CARGO_FEATURE_MIMALLOC_ALLOCATOR").is_ok();
    let use_jemalloc = env::var("CARGO_FEATURE_JEMALLOC_ALLOCATOR").is_ok();
    
    if use_mimalloc {
        println!("cargo:rustc-cfg=using_mimalloc");
        println!("cargo:warning=📦 Using mimalloc allocator");
    } else if use_jemalloc {
        println!("cargo:rustc-cfg=using_jemalloc");
        println!("cargo:warning=📦 Using jemalloc allocator");
    } else {
        println!("cargo:rustc-cfg=using_system_allocator");
        println!("cargo:warning=📦 Using system allocator");
    }
}

/// Configure link-time optimizations and linker settings
fn configure_linking() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let profile = env::var("PROFILE").unwrap_or_default();
    
    // Use LLD linker for faster linking if available
    if has_lld() {
        println!("cargo:rustc-link-arg=-fuse-ld=lld");
        println!("cargo:warning=⚡ Using LLD linker for faster builds");
    }
    
    match target_os.as_str() {
        "linux" => {
            // Linux-specific optimizations
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
            println!("cargo:rustc-link-arg=-lpthread");
            
            if profile == "release" {
                // Strip debug symbols in release
                println!("cargo:rustc-link-arg=-Wl,--strip-debug");
                // Enable GNU_RELRO for security
                println!("cargo:rustc-link-arg=-Wl,-z,relro");
                println!("cargo:rustc-link-arg=-Wl,-z,now");
            }
        }
        "macos" => {
            // macOS-specific optimizations
            println!("cargo:rustc-env=MACOSX_DEPLOYMENT_TARGET=10.14");
            
            if profile == "release" {
                // Dead code stripping
                println!("cargo:rustc-link-arg=-Wl,-dead_strip");
            }
        }
        "windows" => {
            // Windows-specific optimizations
            if profile == "release" {
                // Link-time code generation
                println!("cargo:rustc-link-arg=/LTCG");
                // Optimize for size
                println!("cargo:rustc-link-arg=/OPT:REF");
                println!("cargo:rustc-link-arg=/OPT:ICF");
            }
        }
        _ => {}
    }
}

/// Platform-specific configurations
fn configure_platform() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    
    // Enable platform-specific features
    match target_os.as_str() {
        "linux" => {
            println!("cargo:rustc-cfg=platform_linux");
            
            // Check for musl vs glibc
            if target_env == "musl" {
                println!("cargo:rustc-cfg=platform_musl");
                println!("cargo:warning=📦 Building for musl libc");
            }
        }
        "macos" => {
            println!("cargo:rustc-cfg=platform_macos");
            
            // Check for Apple Silicon
            if env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default() == "aarch64" {
                println!("cargo:rustc-cfg=platform_apple_silicon");
                println!("cargo:warning=🍎 Building for Apple Silicon");
            }
        }
        "windows" => {
            println!("cargo:rustc-cfg=platform_windows");
        }
        _ => {}
    }
}

/// Detect cache line size for optimal alignment
fn detect_cache_line_size() {
    // Default to 64 bytes (common for x86_64)
    let cache_line_size = if cfg!(target_arch = "x86_64") {
        64
    } else if cfg!(target_arch = "aarch64") {
        128 // Some ARM processors have 128-byte cache lines
    } else {
        64 // Safe default
    };
    
    println!("cargo:rustc-env=CACHE_LINE_SIZE={}", cache_line_size);
    println!("cargo:rustc-cfg=cache_line_size_{}", cache_line_size);
}

/// Check if we're cross-compiling
fn is_cross_compiling() -> bool {
    let host = env::var("HOST").unwrap_or_default();
    let target = env::var("TARGET").unwrap_or_default();
    !host.is_empty() && !target.is_empty() && host != target
}

/// Check if LLD linker is available
fn has_lld() -> bool {
    Command::new("ld.lld")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}