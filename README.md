# cloudchamberdemo

You can put a number of videos into a folder, and from which the notebook will extract frames and pass to a trained retinanet model.

The model will recognize any number of tracks from the frame. Overlapping tracks will be screened out and left with the one with the highest score.

This notebook logs all appeared tracks and if another track happens to appear near already existed tracks, they are neglected. A time_scale parameter is used to detemine the existance.

With the log of appeared tracks, the notebook output a table of each type of tracks and their binned count at a time-step manner. You can use this table to find out the change of track appearance over time.
