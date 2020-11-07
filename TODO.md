## Nov.03.2020 summary
- [ ] Change the structure of code, do RetinaNet and SPANet Detection in Real-Time
    - Notation
        - [`startImg`, `actImg`] -> `pix2food` -> `fakeImg`(or pushed image)
        - `startImg`: orginal image from camera
        - `actImg`: generated based on food bbox detected from `startImg`, set some defaults in case no food bbox deteced
        - `fakeImg`: generated from `pix2food`
        - `Real Pushed Image` = next `startImg`, it's the ground truth for `fakeImg`
    - high level thoughts: 
        1. generate 4 actImg and 4 fakeImg(pushed images) in real-time.
            - no matter whether there's food or not (empty plate).
            - no matter whether the food in the plate is detected or not
            - when no food, default 4 actImg will be (center of plate -> top, bot, left, right)
            - 
        2. use RetinaNet and SPANet to detect and predict bbox and acquisition for 5 images(1 `startImg` + 4 `fakeImg`).
            - what will happen if no food or empty plate in `startImg`? the `fakeImg` should be the same as `startImg`
            - However, some artifacts mpotato will show up in fakeImg, due to badly trained `pix2food`(enhance dataset to cover this case)
        3. visualize 5 images(1 `startImg` + 4 `fakeImg`) with corresponding actImg.
        4. choose the one with best Aquisition Action to pub to `Aikido`
            - if it's a `fakeImg` then send the `actImg` to `Ada`
            - if it's the `startImg`, then send `scoop` to `Ada`
        5. based on `fakeImg`, pre plan trajectory for scooping?
            - what if `fakeImg` doesn't match well with `Real Pushed Image`?
        6. redo RetinaNet, SPANet on `Real Pushed Image` then repeat steps above.
    - details:
        - set a callback for camera image topic
        - in the callback, img -> RetinaNet -> bbox -> SPANet -> 4 MarkerArray
        - genrate 4 actImg based on bbox(which bbox TBD) with default actImg(from img center to 4 directions)
        - Pix2Food -> 4 fakeImg -> RetinaNet -> 4 bboxes -> SPANet -> 4 MarkerArray
        - Publish all 4 fakeImg with actImg associated with them
        - it's ok when no food show up or no food detected for pix2food for now, deal with it latter
        - figure out the logic of which marker to pub

- [ ] Debugging ROS Network Setup between `weebo` and `squirrel`(my own desktop)
    - when running only `pix2food` model on ``squirrel` and everything else on `weebo`, there will be some bugs as mentioned below
    - this only happen after I add visualizaiton of [`startImg`, `actImg`, `fakeImg`]
    - ROS Master cannot revceive msg from `reconfig_manager`, `original`, `pushed`  topic and `/food/marker_array` topic

- [ ] Data Collection Improvement
    - Empty case: when no mpotato, whatever action given, the image won't change
    - Reasonable random actImg generator(to generate "whatever action" above).

## Nov.05.2020

- [ ] 

