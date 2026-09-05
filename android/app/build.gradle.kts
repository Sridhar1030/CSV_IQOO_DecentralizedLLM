plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "dev.dllm.node"
    compileSdk = 35

    defaultConfig {
        applicationId = "dev.dllm.node"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
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
