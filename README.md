# Motion Lib
A motion data preprocess repo, reads motion sequence, coordinate system conversion, format conversion, feature extraction

1. Read motion data from different format and save them into a dictionary/Object （倾向于dictionary）
2. Convert motion data
    - Left coordinate system to right coordinate and vise versa
    - y up to z up and vise versa
3. Motion data preprocess
    - convert the poses to relative to root
    - normalize the root starting direction
4. Change Skeleton
