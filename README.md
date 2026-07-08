                                                                          My Doorbell Camera Analysis
For this homework, I wanted to see if I could clear up blurry and/or dark doorbell camera footage to better identify obscured figures. I developed a methodical procedure to modify photos, reproduce a lot of camera angles, and try different approaches to outline shapes and persons, because security cameras frequently have very bad illumination and blurry images.

                                                                            Setup and Execution
It was easy to get going. I made sure that Python and a few common tools for math and image processing were installed on my laptop. Then, I configured my project so my workspace pointed directly to the folder containing my doorbell photos, allowing me to maintain organization and prevent mistakes.

Because each step of my project relies on the pictures created in the step before it, I had to run my work in a strictly sequential order. After collecting the baseline statistics and adjusting the illumination, I adjusted the camera angles and the blurring, and then used the edge-finding tools. I was able to securely convert my single initial picture into a large test batch of more than 200 images by following this procedure.

                                                                            My Process Explained
To make the job manageable, I divided it into four main phases:

Fixing the Lighting: To analyze the raw data, I divided the original image into its primary color layers. I separated only the brightness layer of the image and computationally extended the contrast to correct the harsh shadows and dazzling bright spots you often see outside. This evenly lightened the dark areas without taking away from the colors themselves.

Simulating Camera Angles: I created a loop that zoomed and rotated the picture. This loop gave me 14 different viewpoints by varying the angle and zoom a little bit with each new image, rather than performing the same edit each time.

Testing Blurs: Next, I applied many degrees of a Gaussian blur to create a smoothing effect. I created seven distinct levels of fuzziness to test against, using a straightforward method that automatically matched the size of the computer's "blur window" to the blur's intensity.

Finding Outlines: Finally, I randomly shuffled my blurred images into test groups and stripped away all the color, turning them grayscale. I then ran them through four different mathematical tools (Sobel, Laplacian, Canny, and Prewitt) designed to find the physical outlines of shapes.

                                                                          Gaussian Blur Analysis
I used σ levels of 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, and 3.5 to apply a Gaussian blur to every image. The smoothing effect became more noticeable when I raised the σ value. I discovered that the effect was subtle at lower levels (σ=0.5, 1.0), substantially decreasing small noise while preserving important features. However, by σ=2.5 and higher, the image was dominated by the blur. It reduced the structural clarity of my edge maps and turned intricate textures, such as grass or clothing, into smooth gradients. Higher σ values in binary images led to object outlines expanding and overlapping, which was helpful for identifying shapes in general but masked important details.

                                                                          Edge Detection Analysis
I contrasted four methods, each with unique trade-offs:

Sobel and Prewitt: I found these to be effective for direction-specific gradients, but they frequently produced thick, imprecise margins because they were too sensitive to noise.

Laplacian: Although this made it easier for me to capture small details, it also increased image noise, making it challenging to distinguish actual edges.

Canny: This was the most reliable approach. The cleanest, most continuous edges were produced by its multi-stage technique, which included hysteresis thresholding and noise reduction.

I came to the conclusion that Canny with a pre-applied Gaussian blur is the best option for my doorbell footage, which is frequently hampered by low-light sensor noise. By acting as a crucial pre-processing stage, the blur stops sensor noise from being identified as edges. The primary limitation I faced was balancing noise suppression with feature retention; I learned that aggressive blurring can easily destroy identifying features, so I had to carefully tune σ based on the camera's lighting conditions.

My code is written in Python, primarily relying on OpenCV for image manipulation. First, I used cv2.split() to separate the original image into its Blue, Green, and Red layers to calculate basic pixel statistics. To fix the harsh outdoor lighting, I converted the photo to the HSV color space, targeted just the brightness layer, and used cv2.equalizeHist() to automatically stretch the contrast. This evenly brightened the dark shadows without distorting the actual colors.

To simulate different camera angles, I wrote a loop utilizing cv2.getRotationMatrix2D() and cv2.warpAffine(). By tying the rotation and zoom math directly to the loop count, each of the 14 generated images received a totally unique perspective. For the smoothing phase, I used cv2.GaussianBlur(). Since the computer needs a physical, odd-numbered grid to calculate blurs, I wrote a quick math formula that automatically scaled the grid size perfectly alongside my chosen blur strengths (σ). Finally, I used built-in functions like cv2.Sobel() and cv2.Canny() to trace the physical outlines of the figures. Because OpenCV doesn't have a native Prewitt tool, I manually coded the mathematical 3x3 matrices myself and applied them using cv2.filter2D().

                                                                                      Conclusion
To sum up, our study illustrated the practical benefits of using image processing pipelines to improve security footage. I successfully bridged the gap between raw, noisy data and useful visual information by iterating through a variety of pre-processing techniques, particularly color space normalization, affine transformations, and Gaussian blurring.

According to my investigation, the Canny method is the most dependable for noisy situations like doorbell camera captures, even though edge detection techniques like Sobel, Prewitt, and Laplacian offer several ways to interpret image gradients. This pipeline's performance really depends on how carefully the settings are balanced: too little blurring results in misleading signals, while too much blurring causes me to lose identity-defining features.

In the end, this project gave me a clear roadmap for computer vision tasks: always validate against the particular constraints of the environment, select the appropriate filter for the noise profile, and clean the data first. I now have a reliable, consistent procedure that can be readily modified for more complex uses, such as motion analysis in difficult, low-light situations or automated object tracking.
