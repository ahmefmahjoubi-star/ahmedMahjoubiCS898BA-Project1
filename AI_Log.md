# AI Usage Log
| Date and Time | Prompt | Tool | Response Synopsis | Change |
| :--- | :--- | :--- | :--- | :--- |
| 06/14/2026<br>12:20 AM | "Can you give me the Markdown syntax for a 5-column table to use as a tracking log?" | Gemini | Provided the basic formatting structure for a Markdown table. | Used the template to correctly format the layout of `AI_Log.md`. |
| 06/14/2026<br>08:35 AM | "What is the NumPy function to find the maximum value in an array?" | Gemini | Provided `np.max()` syntax. | Used to calculate channel statistics in `image_stats.py`. |
| 06/14/2026<br>09:30 AM | "What is the OpenCV function for histogram equalization?" | Gemini | Provided usage of `cv2.equalizeHist()`. | Added `image_equalize.py` to the repository. |
| 06/28/2026<br>10:15 AM | "told geminie to improve my grammer and correct all otigraphical mistakes fo rQualitative Analysis" | Gemini | Corrected grammar and spelling errors for the provided text. | Updated Qualitative Analysis documentation with the corrected text. |
| 06/28/2026<br>11:00 AM | "also asked how to Apply mask: keeps color where mask is white | Gemini | Explained how `cv2.bitwise_and` uses a white mask to retain original image colors. | Implemented mask application logic in the image processing script. |
| 07/19/2026<br>01:51 AM | "give me AI coding similar to this for what..." | Gemini | Generated a new Markdown table row matching the existing tracking log format. | Logged the current interaction into `AI_Log.md`. |
| 07/19/2026<br>9:36 AM | "My image preprocessing loop is failing to load the files. Can you find the mistake? There should just be an error in one line." | Gemini | Identified a pathing error and corrected `cv2.imread(img_name)` to the proper `cv2.imread(img_path)`. | Updated the data loading script with the corrected line to successfully read, transpose, and normalize the dataset. |
