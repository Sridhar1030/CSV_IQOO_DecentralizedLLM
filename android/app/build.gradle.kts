plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "dev.dllm.node"
    compileSdk = 35

    defaultConfig {
        applicationId = "dev.dllm.node"
        // 31 is what the Qualcomm LiteRT NPU runtime libraries require; the target phones are API 36.
        minSdk = 31
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
        ndk { abiFilters += "arm64-v8a" }   // the QNN + LiteRT .so ship arm64 only
    }
    // The QNN skel is loaded by the DSP over FastRPC and the dispatch/compiler plugins are dlopen'd
    // by path, so the native libraries must be real files in the app's lib dir, not left compressed
    // inside the APK. Legacy packaging extracts them on install.
    packaging {
        jniLibs {
            useLegacyPackaging = true
            keepDebugSymbols += "**/*.so"
        }
    }
    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    // The only dependency. WebSocket client plus streaming downloads. Apache-2.0.
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
}
