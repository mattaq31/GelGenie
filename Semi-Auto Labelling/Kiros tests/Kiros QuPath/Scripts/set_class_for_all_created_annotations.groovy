def gelBand = getPathClass('Gel Band')
getAnnotationObjects().eachWithIndex { annotation , i ->
      annotation.setPathClass(gelBand)
}
fireHierarchyUpdate()

//To Ignore the rectangular annotation of whole image (which is automatically locked)

def background = getPathClass('Background')
getAnnotationObjects().eachWithIndex { annotation , i ->
    if (annotation.isLocked())
        annotation.setPathClass(background)
}
fireHierarchyUpdate()