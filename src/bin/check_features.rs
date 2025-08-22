fn main() {
    println!("🔍 CPU Feature Detection");
    println!("─────────────────────────");
    
    #[cfg(has_avx512)]
    println!("✅ AVX-512: Enabled");
    #[cfg(not(has_avx512))]
    println!("❌ AVX-512: Not available");
    
    #[cfg(has_avx2)]
    println!("✅ AVX2: Enabled");
    #[cfg(not(has_avx2))]
    println!("❌ AVX2: Not available");
    
    #[cfg(has_fma)]
    println!("✅ FMA: Enabled");
    #[cfg(not(has_fma))]
    println!("❌ FMA: Not available");
    
    #[cfg(has_sse42)]
    println!("✅ SSE4.2: Enabled");
    #[cfg(not(has_sse42))]
    println!("❌ SSE4.2: Not available");
    
    #[cfg(has_popcnt)]
    println!("✅ POPCNT: Enabled");
    #[cfg(not(has_popcnt))]
    println!("❌ POPCNT: Not available");
    
    #[cfg(has_neon)]
    println!("✅ NEON: Enabled (ARM)");
    
    #[cfg(using_mimalloc)]
    println!("📦 Allocator: mimalloc");
    #[cfg(using_jemalloc)]
    println!("📦 Allocator: jemalloc");
    #[cfg(using_system_allocator)]
    println!("📦 Allocator: system");
    
    if let Ok(cores) = std::env::var("CPU_CORES") {
        println!("🔧 CPU Cores: {}", cores);
    }
    
    if let Ok(cache_line) = std::env::var("CACHE_LINE_SIZE") {
        println!("📏 Cache Line: {} bytes", cache_line);
    }
}
