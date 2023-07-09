(From Hongfan:)

Original source files are located at:

`/data/Simulations/chenhf/CME/CR*/StatsAnalysis/cr*_stacked_edge.npy` and
`/data/Simulations/chenhf/CME/CR*/StatsAnalysis/CR*_SimID4edge.npy`


it should be a 90(90 time point) by angle by number of successfulRuns matrix

`CR*SimID4edge.npy` saves the ID of successful runs, so `stacked_edge[:,:,0]` gives you `SimID4edge[0]`

for each run, you should only trust the data before it reaches the boundary. So a simple test can be, check row by row and see which row has maximal value larger than 120(maximal Height is 128, so 120 is fairly close. Don't use something like 125 or 127, as some runs are extremely fast and you may not have a chance to catch them right at the edge.).

also, for some slow runs, edge = 0 do not necessary means that velocity is 0, it is just that it hasn't entered into the field of view. So if you are interested in the velocity, you have to find at least two points that are non-zero.

By the way, here is the function that is dual to function Normalizaiton

```
def Polar_to_Cartesian(edge, start_angle, end_angle, height, width, circles_disk, circles_scope):
    theta = np.arange(0, 2 * np.pi, 2 * np.pi / width)[start_angle:end_angle]
    # Coordinates of disk and telescope center
    circle_x = circles_disk[0]
    circle_y = circles_disk[1]
    # Radius of disk and telescope
    r_disk = circles_disk[2]
    r_scope = circles_scope[2]
    Xp = circle_x + r_disk * np.cos(theta + 1.2 * np.pi)
    Yp = circle_y + r_disk * np.sin(theta + 1.2 * np.pi)
    Xi = circle_x + r_scope * np.cos(theta + 1.2 * np.pi)
    Yi = circle_y + r_scope * np.sin(theta + 1.2 * np.pi)
    r = edge / height
    X = Xp + ( Xi - Xp ) * r
    Y = Yp + ( Yi - Yp ) * r
    return (X,Y)
```



```
## Function: Change of coordinates. From Cartesian to Polar.\n",
    "##\n",
    "## Argument:\n",
    "## image: a nxn numpy array\n",
    "## height and width: the size of transformed image is (height x width)\n",
    "## circles_disk: three dimensional tuple (x,y,r), x,y is the coordinates of the center of\n",
    "##     occulting disk, r is the radius of the occulting disk\n",
    "## circles_scope: three dimensional tuple (x,y,r), x,y is the coordinates of the center of\n",
    "##     instrument, r is the raius of the instrument\n",
    "##\n",
    "## Return Value:\n",
    "## A height x width numpy array.\n",
    "##\n",
    "## Note: circles_disk, circles_scope are true coordinates(origin at bottom left), not index!!!!\n",
    "## Example: the coordinates of x here\n",
    "##[.........]\n",
    "##[..x......]\n",
    "##[.........]\n",
    "##[.........] should be (2,2) but not (1,2)!(0-indexed)\n",
    "## For CR2161, we suggest using theta + pi.\n",
    "## For CR2154, we suggest using theta - pi / 2\n",
    "## so that the images are centred and we can avoid the edge effect.\n",
    "def Normalization(image, height, width, circles_disk, circles_scope):\n",
    "    theta = np.arange(0, 2 * np.pi, 2 * np.pi / width) \n",
    "    # Coordinates of disk and telescope center\n",
    "    circle_x = circles_disk[0]\n",
    "    circle_y = circles_disk[1]\n",
    "    # Radius of disk and telescope\n",
    "    r_disk = circles_disk[2]\n",
    "    r_scope = circles_scope[2]\n",
    "    j,i = np.ix_(np.arange(height),np.arange(width))\n",
    "    r = j/height\n",
    "    Xp = circle_x + r_disk * np.cos(theta + 1.2 * np.pi)\n",
    "    Yp = circle_y + r_disk * np.sin(theta + 1.2 * np.pi)\n",
    "    Xi = circle_x + r_scope * np.cos(theta + 1.2 * np.pi)\n",
    "    Yi = circle_y + r_scope * np.sin(theta + 1.2 * np.pi)\n",
    "    X = Xp + ( Xi - Xp ) * r\n",
    "    Y = Yp + ( Yi - Yp ) * r\n",
    "    shapes = image.shape\n",
    "    int_Y = (shapes[0]-Y).astype(\"int\").reshape(height*width)\n",
    "    int_Y = np.where(int_Y >= shapes[0], shapes[0]-1, int_Y)\n",
    "    int_X = X.astype(\"int\").reshape(height*width)\n",
    "    int_X = np.where(int_X >= shapes[1], shapes[1]-1, int_X)\n",
    "    return image[int_Y, int_X].reshape(height, width).astype(\"float64\")## so basically the rectangle is upside down, and \n",
    "                                ## we should set origin = \"lower\" when visualizing normalized data\n"
```