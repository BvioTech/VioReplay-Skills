fn main() {
    // Link against macOS frameworks needed for accessibility and event tap
    println!("cargo:rustc-link-lib=framework=ApplicationServices");
    println!("cargo:rustc-link-lib=framework=CoreFoundation");
    println!("cargo:rustc-link-lib=framework=CoreGraphics");
    // Vision framework for OCR fallback (macOS 10.15+)
    println!("cargo:rustc-link-lib=framework=Vision");
}
