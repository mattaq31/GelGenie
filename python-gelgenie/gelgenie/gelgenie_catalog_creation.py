from extension_catalog_model.model import *

version_range = VersionRange(min="v1.1.0")

release_new = Release(
   name="v1.1.0",
   main_url="https://github.com/mattaq31/GelGenie/releases/download/v1.1.0/qupath-gelgenie-1.1.0.jar",
   version_range=version_range
)
extension = Extension(
   name="QuPath GelGenie extension",
   description="GelGenie extension that enables the automatic segmentation and analysis of gel images in QuPath.",
   author="Matthew Aquilina",
   homepage="https://github.com/mattaq31/GelGenie",
   releases=[release_new]
)

catalog = Catalog(
   name="GelGenie catalog",
   description="Extensions for the GelGenie automatic gel analysis platform.",
   extensions=[extension]
)

with open("/Users/matt/Desktop/catalog.json", "w") as file:
   file.write(catalog.model_dump_json(indent=2))
   print(file.name + " written")
