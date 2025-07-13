plugins {
    // Support writing the extension in Groovy (remove this if you don't want to)
    groovy
    // To optionally create a shadow/fat jar that bundle up any non-core dependencies
    id("com.gradleup.shadow") version "8.3.5"
    // QuPath Gradle extension convention plugin
    id("qupath-conventions")
}

// TODO: Configure your extension here (please change the defaults!)
qupathExtension {
    name = "qupath-extension-gelgenie"
    group = "io.github.mattaq31"
    version = "1.1.0"
    description = "QuPath extension to directly run and interface with GelGenie models."
    automaticModule = "io.github.mattaq31.extension.gelgenie"
}

// TODO: Define your dependencies here
dependencies {
    shadow(libs.bundles.qupath)
    shadow(libs.bundles.logging)
    shadow(libs.extensionmanager)

    implementation(libs.bundles.markdown)


    testImplementation(libs.bundles.qupath)
    testImplementation(libs.junit)

    shadow(libs.qupath.fxtras) // required for native filechoosers
    implementation(libs.bundles.djl)
}