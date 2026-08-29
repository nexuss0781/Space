# Vision-Benchmark Report (all candidates)

Consolidated comparison of every vision candidate benchmarked on the same
10-image task set with llama.cpp (`llama-mtmd-cli`, build from source).

| Candidate | Model file | Weights | Avg tok/s | Wall (10 img) | Tasks |
|---|---|---:|---:|---:|---:|
| gemma3-4b | gemma-3-4b-it-Q4_K_M.gguf | 2.49 GB | 13.19 | 602.62s | 10 |
| moondream2 | moondream2-050824-q8.gguf | 1.51 GB | 14.86 | 236.07s | 10 |
| smolvlm-500m | SmolVLM-500M-Instruct-Q8_0.gguf | 0.44 GB | 65.06 | 77.64s | 10 |
| smolvlm2-2.2b | SmolVLM2-2.2B-Instruct-Q4_K_M.gguf | 1.11 GB | 19.15 | 363.29s | 10 |
| smolvlm2-500m | SmolVLM2-500M-Video-Instruct-Q8_0.gguf | 0.44 GB | 67.15 | 73.74s | 10 |

## gemma3-4b

- Source: `ggml-org/gemma-3-4b-it-GGUF`
- Files: `gemma-3-4b-it-Q4_K_M.gguf` (2.49 GB) + `mmproj-model-f16.gguf`
- System: x86_64, 4 cores, 16.77 GB RAM
- Wall (10 tasks): 602.62s | avg throughput: 13.19 tok/s

| # | Image | Out tok | eval (ms) | tok/s | peak RSS (MB) | wall (s) |
|---|---|---:|---:|---:|---:|---:|
| 1 | images/01.jpg | 127 | 9670.7 | 13.13 | 5300.82 | 61.13 |
| 2 | images/02.jpg | 127 | 9657.0 | 13.15 | 5300.92 | 61.151 |
| 3 | images/03.jpg | 127 | 9731.5 | 13.05 | 5301.0 | 61.155 |
| 4 | images/04.jpg | 127 | 9569.2 | 13.27 | 5300.84 | 60.781 |
| 5 | images/05.jpg | 127 | 9563.6 | 13.28 | 5300.85 | 61.263 |
| 6 | images/06.jpg | 127 | 9586.4 | 13.25 | 5300.89 | 60.795 |
| 7 | images/07.jpg | 127 | 9813.5 | 12.94 | 5300.78 | 61.148 |
| 8 | images/08.jpg | 111 | 8342.8 | 13.3 | 5309.32 | 59.456 |
| 9 | images/09.jpg | 127 | 9618.5 | 13.2 | 5309.39 | 60.955 |
| 10 | images/10.png | 46 | 3442.4 | 13.36 | 5309.31 | 54.789 |

### Responses

**Task 1 (images/01.jpg):** Describe what you see in this image in detail....

```text
Okay, here's a detailed description of what I see in the image: **Overall Impression:** The image presents a highly stylized and somewhat cluttered desktop setup, likely belonging to a creative professional or someone who enjoys a personalized workspace. The lighting is dramatic, with a strong shadow across the right side of the desk, creating a moody and focused atmosphere. **Desk and Surface:** * **Desk Surface:** The desk itself is a light gray, likely laminate or a similar material. It appears to be a large, rectangular surface. * **Wooden Shelf:** A dark, warm-toned wooden shelf runs acro
```

**Task 2 (images/02.jpg):** What is happening in this urban scene? Describe it....

```text
Here's a description of what's happening in the urban scene depicted in the image: **Overall Impression:** The image evokes a sense of quiet solitude and the transition of time, likely at dusk or dawn. It's a contemplative scene with a focus on the interplay of light and shadow. **Key Elements:** * **Window and Reflection:** The dominant feature is a large window, likely in a building, with a blurred reflection of the outside. The reflection shows a table, possibly a dining table, and a sheer white curtain. The color of the reflection suggests it's either early morning or late evening,
```

**Task 3 (images/03.jpg):** Describe the subject and setting of this image....

```text
Here's a description of the subject and setting in the image: **Subject:** The primary subject is a range of mountains, likely part of a larger mountain system. The image focuses on the layered effect of the peaks receding into the distance. **Setting:** * **Landscape:** The scene is a mountainous region, characterized by steep slopes and valleys. The mountains appear rugged and somewhat dark, suggesting a cool, possibly overcast, atmosphere. * **Atmosphere:** There's a significant amount of atmospheric haze or mist in the distance, creating a soft, diffused light and a sense of depth. This ha
```

**Task 4 (images/04.jpg):** What objects and colors are visible in this image?...

```text
Here's a breakdown of the objects and colors visible in the image: **Objects:** * **Pond/Muddy Patch:** A large, irregular puddle of water and mud dominates the foreground. * **Road:** A paved road runs along the right side of the image, curving slightly. * **Grass/Vegetation:** Dry, brown grass and other vegetation are present along the edges of the road and in the background. There are some taller, grassy clumps. * **Hill/Slope:** A grassy hill or slope rises in the background. * **Trees:** A few small
```

**Task 5 (images/05.jpg):** Describe the scenery and atmosphere in this image....

```text
Here's a description of the scenery and atmosphere in the image: **Scenery:** * **Lake Scene:** The dominant feature is a calm, expansive lake. The water is a muted, silvery-gray, reflecting the sky. * **Distant Hills:** In the far distance, a line of low, hazy hills or mountains is visible, adding depth to the scene. They are silhouetted against the sky. * **Wooden Dock:** The foreground is a weathered wooden dock, likely a pier or small jetty extending into the lake. The wood has a warm, reddish-brown tone.
```

**Task 6 (images/06.jpg):** What does this image show? Give a detailed description....

```text
Here's a detailed description of the image: **Overall Impression:** The image is a close-up shot of a richly colored, patterned blanket or shawl draped over a wooden surface. The lighting is warm and slightly muted, creating a cozy and inviting feel. **Detailed Description:** * **The Blanket/Shawl:** The primary subject is a large blanket or shawl with a complex, geometric pattern. The design is predominantly composed of triangles and diamonds in a vibrant palette of colors. * **Colors:** The colors are intense and saturated, including shades of red, orange, pink, turquoise, teal,
```

**Task 7 (images/07.jpg):** Describe the composition and content of this image....

```text
Here's a description of the composition and content of the image: **Overall Impression:** The image is a striking, atmospheric photograph of a canal in Hamburg, Germany. It’s dominated by strong vertical lines and a dramatic use of light and shadow, creating a sense of depth and grandeur. **Composition:** * **Leading Lines:** The most dominant compositional element is the strong, converging set of vertical lines created by the buildings on either side of the canal. These lines powerfully draw the viewer's eye into the distance, towards the bright light at the end of the canal. * **Rule of Thir
```

**Task 8 (images/08.jpg):** What is the main subject, and what surrounds it?...

```text
Here's an analysis of the image: * **Main Subject:** The main subject is a bunch of dark purple grapes held in a pair of hands. * **What Surrounds It:** The grapes are surrounded by a man's hands, which are outstretched and holding the grapes. The man is wearing a light blue shirt. The background is blurred and dark, suggesting an outdoor setting, possibly a vineyard or garden. Let me know if you'd like me to describe any specific aspect of the image in more detail!
```

**Task 9 (images/09.jpg):** Describe what this image depicts in full detail....

```text
Here's a detailed description of the image: **Overall Impression:** The image depicts a vast, snow-covered evergreen forest shrouded in a thick, heavy mist or fog. It evokes a sense of isolation, stillness, and the raw beauty of a winter landscape. **Foreground:** * **Trees:** The dominant feature is a dense, tightly packed forest of tall, coniferous trees. They appear to be primarily fir trees, judging by their shape and needle-like foliage. * **Snow Cover:** The trees are completely blanketed in snow, with a thick layer of white covering their branches and trunks. The
```

**Task 10 (images/10.png):** Read all the text visible in this image and transcribe it ex...

```text
Here's the transcription of all the text visible in the image: **STOP** **CAFE OPEN** Welcome to the New World of AGI. Speed matters = micro = 360M.
```


---

## moondream2

- Source: `cjpais/moondream2-llamafile`
- Files: `moondream2-050824-q8.gguf` (1.51 GB) + `moondream2-mmproj-050824-f16.gguf`
- System: x86_64, 4 cores, 16.77 GB RAM
- Wall (10 tasks): 236.07s | avg throughput: 14.86 tok/s

| # | Image | Out tok | eval (ms) | tok/s | peak RSS (MB) | wall (s) |
|---|---|---:|---:|---:|---:|---:|
| 1 | images/01.jpg | 90 | 6151.4 | 14.63 | 3991.86 | 24.936 |
| 2 | images/02.jpg | 102 | 7042.2 | 14.48 | 3991.97 | 25.753 |
| 3 | images/03.jpg | 75 | 5053.0 | 14.84 | 3991.97 | 23.71 |
| 4 | images/04.jpg | 61 | 4131.9 | 14.76 | 3990.33 | 22.746 |
| 5 | images/05.jpg | 95 | 6446.3 | 14.74 | 3993.93 | 25.103 |
| 6 | images/06.jpg | 100 | 6869.1 | 14.56 | 3990.38 | 25.5 |
| 7 | images/07.jpg | 56 | 3746.9 | 14.95 | 3992.07 | 22.388 |
| 8 | images/08.jpg | 41 | 2701.8 | 15.18 | 4000.05 | 21.394 |
| 9 | images/09.jpg | 83 | 5541.1 | 14.98 | 4000.02 | 24.158 |
| 10 | images/10.png | 25 | 1614.3 | 15.49 | 4000.04 | 20.381 |

### Responses

**Task 1 (images/01.jpg):** Describe what you see in this image in detail....

```text
The image shows a well-organized workspace with a large computer monitor on the left side of the frame, a keyboard and mouse on the right side, and a coffee cup and mug on the left side. The desk is made of wood and has a wooden shelf above it, which holds various items such as a small figurine, a plant, and a book. The background is a white wall, providing a clean and minimalist backdrop for the workspace.
```

**Task 2 (images/02.jpg):** What is happening in this urban scene? Describe it....

```text
The image shows a wooden chair sitting in front of a window. The window is divided into two sections, with one section being covered by a white curtain and the other section being left open. The chair is positioned in front of the open window, creating a contrast between the interior and exterior spaces. The scene appears to be set in an urban environment, as there are no visible signs of a city or buildings. The chair and window are the main elements in the image, with no other objects or people visible.
```

**Task 3 (images/03.jpg):** Describe the subject and setting of this image....

```text
The image shows a beautiful mountain range with a hazy blue sky in the background. The mountains are covered in a thick layer of fog, creating a serene and peaceful atmosphere. The image is taken from a high vantage point, providing a panoramic view of the landscape. The colors in the image are predominantly blue and gray, reflecting the foggy and mountainous environment.
```

**Task 4 (images/04.jpg):** What objects and colors are visible in this image?...

```text
The image features a dirt road with tire tracks, a body of water, and a sunset in the background. The road has tire tracks, indicating that vehicles have recently passed through the area. The body of water is surrounded by tall grass, and the sunset casts a warm, golden glow over the landscape.
```

**Task 5 (images/05.jpg):** Describe the scenery and atmosphere in this image....

```text
The image shows a serene scene of a wooden dock extending into a body of water. The dock is equipped with a metal railing, and there is a small bird perched on the railing. The water appears calm and tranquil, with no visible boats or people. The sky above is a beautiful shade of blue, suggesting a clear and sunny day. The overall atmosphere of the image is peaceful and inviting, with the dock and its surroundings providing a perfect spot for relaxation and contemplation.
```

**Task 6 (images/06.jpg):** What does this image show? Give a detailed description....

```text
The image shows a close-up view of a colorful, patterned blanket or shawl. The blanket has a vibrant mix of colors, including shades of pink, blue, and green, and features a unique geometric pattern. The blanket is folded neatly, with the edges of the blanket slightly curled, giving it a sense of depth and dimension. The blanket is placed on a wooden surface, possibly a table or a bench, which adds to the overall rustic and cozy atmosphere of the scene.
```

**Task 7 (images/07.jpg):** Describe the composition and content of this image....

```text
The image shows a boat traveling down a river in a city. The river is surrounded by tall buildings, and a bridge can be seen in the background. The scene appears to be captured during the day, with the sun shining brightly and casting light on the water and the buildings.
```

**Task 8 (images/08.jpg):** What is the main subject, and what surrounds it?...

```text
The main subject is a bunch of grapes, and it is being held in the hands of a person. The grapes are surrounded by the person's fingers, creating a close-up image of the grapes.
```

**Task 9 (images/09.jpg):** Describe what this image depicts in full detail....

```text
The image shows a dense forest of snow-covered pine trees. The trees are tall and have a conical shape, with their branches covered in a thick layer of snow. The forest appears to be located in a mountainous area, as the trees are situated at varying heights and the landscape is covered in snow. The image is taken from a high vantage point, providing a comprehensive view of the forest and its surroundings.
```

**Task 10 (images/10.png):** Read all the text visible in this image and transcribe it ex...

```text
If you have any questions or need further information, please feel free to reach out to me. I'm happy to help!
```


---

## smolvlm-500m

- Source: `ggml-org/SmolVLM-500M-Instruct-GGUF`
- Files: `SmolVLM-500M-Instruct-Q8_0.gguf` (0.44 GB) + `mmproj-SmolVLM-500M-Instruct-f16.gguf`
- System: x86_64, 4 cores, 16.77 GB RAM
- Wall (10 tasks): 77.64s | avg throughput: 65.06 tok/s

| # | Image | Out tok | eval (ms) | tok/s | peak RSS (MB) | wall (s) |
|---|---|---:|---:|---:|---:|---:|
| 1 | images/01.jpg | 127 | 1993.9 | 63.7 | 1025.8 | 8.269 |
| 2 | images/02.jpg | 127 | 1982.2 | 64.07 | 1025.68 | 8.119 |
| 3 | images/03.jpg | 127 | 2050.1 | 61.95 | 1025.89 | 8.187 |
| 4 | images/04.jpg | 127 | 2000.1 | 63.5 | 1025.74 | 8.169 |
| 5 | images/05.jpg | 127 | 1987.2 | 63.91 | 1025.58 | 8.144 |
| 6 | images/06.jpg | 127 | 2043.8 | 62.14 | 1025.55 | 8.221 |
| 7 | images/07.jpg | 127 | 1942.7 | 65.37 | 1025.64 | 8.092 |
| 8 | images/08.jpg | 8 | 116.1 | 68.91 | 1025.41 | 6.208 |
| 9 | images/09.jpg | 127 | 1920.3 | 66.13 | 1025.8 | 8.038 |
| 10 | images/10.png | 7 | 98.6 | 70.97 | 1025.62 | 6.193 |

### Responses

**Task 1 (images/01.jpg):** Describe what you see in this image in detail....

```text
The image depicts a cluttered desk setup, showcasing various computer peripherals and accessories. The desk is made of a dark wood or wood-like material, giving it a rustic and vintage appearance. The desk is divided into several sections, each containing a different piece of computer hardware or accessory. 1. **Keyboard and Mouse**: - **Keyboard**: Positioned centrally on the desk. The keyboard is a standard QWERTY layout with a white color. - **Mouse**: Located to the right of the keyboard. The mouse is a standard mouse with a white color and a scroll wheel. 2. **
```

**Task 2 (images/02.jpg):** What is happening in this urban scene? Describe it....

```text
The image depicts a serene outdoor scene with a focus on a window and its surroundings. The window is framed by two wooden posts, each adorned with a dark brown color, suggesting a rustic or traditional design. The window is covered with a sheer white curtain that partially obscures the view outside. The curtain is slightly wrinkled, indicating that it has been used recently. In front of the window, there is a wooden trash can. The trash can is painted in a light blue color and has a handle at the top for easy access. The trash can is positioned in front of the window, providing a clear view o
```

**Task 3 (images/03.jpg):** Describe the subject and setting of this image....

```text
The image depicts a mountainous landscape under a clear sky. The mountains are depicted in varying shades of blue and gray, with the peaks and valleys creating a rugged and imposing appearance. The sky is a gradient of light blue at the top and transitions to a darker blue near the horizon, indicating that the sun is either setting or rising. The light filtering through the clouds creates a soft, ethereal glow on the mountains, casting long shadows that accentuate the contours of the peaks and valleys. The mountains are covered with a variety of vegetation, including some sparse trees and shru
```

**Task 4 (images/04.jpg):** What objects and colors are visible in this image?...

```text
The image depicts a landscape scene dominated by natural elements. The primary focus is on a road that runs through the center of the image. The road is unpaved and appears to be made of dirt or gravel, with some patches of grass and vegetation visible along its edges. The road is bordered by a flat, uneven terrain that includes a mix of dirt, gravel, and possibly some rocks or small hills. In the background, there is a series of hills or mountains that rise up towards the horizon. These hills are covered with a mix of vegetation, including some trees and bushes, suggesting a temperate or cool
```

**Task 5 (images/05.jpg):** Describe the scenery and atmosphere in this image....

```text
The image depicts a serene coastal scene. The foreground is dominated by a wooden dock extending into the calm waters of a large body of water. The dock is weathered and has a weathered appearance, suggesting it has been in use for some time. The wooden planks are slightly worn, indicating that they have been in use for a considerable period. The water in the background is relatively calm, with gentle waves lapping at the shore. The horizon is dotted with rolling hills, which are silhouetted against the sky. The sky is clear, with a soft gradient from light blue at the top to a deeper blue nea
```

**Task 6 (images/06.jpg):** What does this image show? Give a detailed description....

```text
The image depicts a person wearing a colorful, patterned sweater with a fringe border. The sweater has a wide, open collar and is predominantly composed of a multicolored pattern, which appears to be made up of various shades of blue, green, yellow, and pink. The pattern is intricate and seems to have a repetitive design, giving the sweater a somewhat abstract and artistic look. The sweater is draped over the person's shoulders and upper arms, with the fringe border extending outwards from the collar and down the sleeves. The person is standing against a wooden surface, which appears to be a b
```

**Task 7 (images/07.jpg):** Describe the composition and content of this image....

```text
The image depicts a narrow canal in a city, with buildings lining both sides of the waterway. The canal is calm and reflective, with the water appearing to be a light blue color. The buildings on both sides of the canal are tall and have a variety of architectural styles, including some with intricate stonework and others with more modern facades. The buildings are primarily made of brick and have a mix of rectangular and square shapes. The canal itself is bordered by a bridge, which is visible in the distance. The bridge is wide enough for a few vehicles to pass through, and it appears to be 
```

**Task 8 (images/08.jpg):** What is the main subject, and what surrounds it?...

```text
There is a person in the image.
```

**Task 9 (images/09.jpg):** Describe what this image depicts in full detail....

```text
The image depicts a dense forest of evergreen trees, likely coniferous, under a foggy or misty sky. The trees are tall and straight, with dark green needles that appear to be coated with a light dusting of snow. The snow on the trees is thick, covering the branches and needles, creating a stark contrast against the dark green of the trees. The trees are densely packed, with some branches appearing to be obscured by the snow. The background of the image is a pale gray sky, which suggests that the fog or mist is thick enough to obscure the details of the sky. The sky is not very clear,
```

**Task 10 (images/10.png):** Read all the text visible in this image and transcribe it ex...

```text
STOP CAFE OPEN
```


---

## smolvlm2-2.2b

- Source: `ggml-org/SmolVLM2-2.2B-Instruct-GGUF`
- Files: `SmolVLM2-2.2B-Instruct-Q4_K_M.gguf` (1.11 GB) + `mmproj-SmolVLM2-2.2B-Instruct-f16.gguf`
- System: x86_64, 4 cores, 16.77 GB RAM
- Wall (10 tasks): 363.29s | avg throughput: 19.15 tok/s

| # | Image | Out tok | eval (ms) | tok/s | peak RSS (MB) | wall (s) |
|---|---|---:|---:|---:|---:|---:|
| 1 | images/01.jpg | 127 | 6410.4 | 19.81 | 4258.86 | 37.308 |
| 2 | images/02.jpg | 127 | 6416.2 | 19.79 | 4259.0 | 37.03 |
| 3 | images/03.jpg | 126 | 6363.7 | 19.8 | 4259.04 | 37.138 |
| 4 | images/04.jpg | 127 | 6542.3 | 19.41 | 4258.88 | 37.282 |
| 5 | images/05.jpg | 123 | 6671.0 | 18.44 | 4259.12 | 37.464 |
| 6 | images/06.jpg | 127 | 6565.1 | 19.34 | 4258.99 | 37.367 |
| 7 | images/07.jpg | 127 | 6722.8 | 18.89 | 4258.92 | 37.745 |
| 8 | images/08.jpg | 19 | 1194.7 | 15.9 | 4259.07 | 32.487 |
| 9 | images/09.jpg | 127 | 6405.4 | 19.83 | 4259.02 | 37.502 |
| 10 | images/10.png | 25 | 1234.3 | 20.25 | 4260.09 | 31.966 |

### Responses

**Task 1 (images/01.jpg):** Describe what you see in this image in detail....

```text
The image depicts a well-organized and aesthetically pleasing workspace setup on a wooden desk. The desk is equipped with various items that contribute to a professional yet comfortable environment. Starting from the top, there is a large monitor positioned centrally on the desk. The monitor is flanked by two speakers, one on each side, which are strategically placed to provide balanced sound coverage. The speakers are white and appear to be of a modern design, enhancing the professional look of the setup. To the left of the monitor, there is a white keyboard, which is placed directly in front
```

**Task 2 (images/02.jpg):** What is happening in this urban scene? Describe it....

```text
The image captures a serene urban scene, dominated by a wooden chair that stands out against the backdrop of a large window. The chair, with its simple design, is positioned on a balcony, offering a view of the city below. The window, adorned with white curtains, is open, allowing a glimpse of the cityscape. The city lights are visible in the distance, adding a touch of vibrancy to the otherwise tranquil setting. The image is taken from a low angle, making the chair appear larger and more prominent in the frame. The overall composition of the image suggests a peaceful coexistence between the u
```

**Task 3 (images/03.jpg):** Describe the subject and setting of this image....

```text
The image captures a serene landscape of mountains under a clear sky. The mountains, with their dark silhouettes, are densely packed and appear to be of varying heights. They are covered in a blanket of mist, which adds a sense of depth and mystery to the scene. The misty mountains are set against a backdrop of a clear sky, which is lightly colored with a few clouds. The overall atmosphere is tranquil and peaceful, with the mountains providing a stark contrast to the sky. The image does not contain any discernible text or specific actions being performed by objects. The focus is solely on the 
```

**Task 4 (images/04.jpg):** What objects and colors are visible in this image?...

```text
The image captures a serene landscape bathed in the warm glow of the setting sun. The sun, low on the horizon, casts a golden hue over the entire scene. The sky, a canvas of deep blue, is dotted with wispy clouds that add to the tranquil ambiance. The foreground is dominated by a dirt road, its surface strewn with patches of snow. The snow, pristine and untouched, adds a touch of winter to the otherwise summer scene. It's as if the sun's rays are melting the snow, creating a beautiful pattern on the road. The background is a mix of lush green trees and
```

**Task 5 (images/05.jpg):** Describe the scenery and atmosphere in this image....

```text
The image captures a serene lakeside scene during sunset. The wooden dock, constructed from planks, extends into the tranquil water, leading to a metal ladder with two curved handles. The dock's wooden planks are weathered, adding a rustic charm to the scene. The water, calm and reflective, mirrors the sky above. The sky, painted in hues of orange and yellow, suggests the sun is setting, casting a warm glow over the entire scene. In the distance, the silhouette of distant mountains adds depth to the landscape. The overall atmosphere is peaceful and idyllic, evoking a sense of tranquility and s
```

**Task 6 (images/06.jpg):** What does this image show? Give a detailed description....

```text
The image captures a close-up view of a vibrant, multicolored shawl draped over a wooden bench. The shawl, adorned with a geometric pattern in shades of pink, blue, and green, is fringed with blue tassels that add a touch of elegance to its design. The wooden bench, with its natural grain and texture, provides a stark contrast to the colorful shawl, creating a visually appealing juxtaposition. The background is blurred, drawing focus to the shawl and the bench, and suggesting an outdoor setting. The overall image exudes a sense of warmth and comfort,
```

**Task 7 (images/07.jpg):** Describe the composition and content of this image....

```text
The image depicts a serene and picturesque scene of a canal in a European city. The canal, which is the central focus of the image, is flanked by tall, red brick buildings on both sides. These buildings, with their numerous windows, add a sense of architectural grandeur to the scene. The buildings are evenly spaced along the canal, creating a harmonious balance. In the middle of the image, a boat is visible on the water, adding a dynamic element to the otherwise static scene. The boat is moving towards the right side of the image, suggesting a sense of movement and life in the otherwise tranqu
```

**Task 8 (images/08.jpg):** What is the main subject, and what surrounds it?...

```text
The main subject is a bunch of grapes, and it is surrounded by a person's hands.
```

**Task 9 (images/09.jpg):** Describe what this image depicts in full detail....

```text
The image presents a serene and expansive view of a snow-covered forest. The forest is densely populated with tall evergreen trees, their branches heavy with snow. The snow blankets the trees, creating a white, wintry landscape that dominates the foreground of the image. The trees are arranged in a somewhat haphazard manner, with some trees standing taller and more prominently than others, adding to the overall depth and complexity of the scene. The forest extends into the background, where the trees become smaller and more spaced out, gradually giving way to a foggy, misty atmosphere. This fo
```

**Task 10 (images/10.png):** Read all the text visible in this image and transcribe it ex...

```text
STOP Cafe Open Welcome to the New World of AGI. speed matters = micro = 360M.
```


---

## smolvlm2-500m

- Source: `ggml-org/SmolVLM2-500M-Video-Instruct-GGUF`
- Files: `SmolVLM2-500M-Video-Instruct-Q8_0.gguf` (0.44 GB) + `mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf`
- System: x86_64, 4 cores, 16.77 GB RAM
- Wall (10 tasks): 73.74s | avg throughput: 67.15 tok/s

| # | Image | Out tok | eval (ms) | tok/s | peak RSS (MB) | wall (s) |
|---|---|---:|---:|---:|---:|---:|
| 1 | images/01.jpg | 127 | 2172.6 | 58.45 | 1025.41 | 8.398 |
| 2 | images/02.jpg | 9 | 126.5 | 71.15 | 1025.91 | 5.982 |
| 3 | images/03.jpg | 127 | 1926.6 | 65.92 | 1025.52 | 7.806 |
| 4 | images/04.jpg | 127 | 1946.7 | 65.24 | 1025.82 | 7.851 |
| 5 | images/05.jpg | 127 | 1923.8 | 66.01 | 1025.7 | 7.807 |
| 6 | images/06.jpg | 127 | 1936.8 | 65.57 | 1025.46 | 7.884 |
| 7 | images/07.jpg | 127 | 1907.5 | 66.58 | 1025.64 | 7.82 |
| 8 | images/08.jpg | 21 | 286.1 | 73.4 | 1025.55 | 6.143 |
| 9 | images/09.jpg | 127 | 1914.2 | 66.35 | 1025.71 | 7.831 |
| 10 | images/10.png | 26 | 357.1 | 72.82 | 1025.87 | 6.218 |

### Responses

**Task 1 (images/01.jpg):** Describe what you see in this image in detail....

```text
The image presents a top-down view of a desk setup, showcasing various objects and their arrangement. The desk, made of wood, features a large, white computer monitor, a keyboard, and a mouse placed on a mouse pad. The monitor is positioned centrally, with the keyboard and mouse to its right. To the left of the monitor, there's a white coffee mug, a small figurine, and a decorative item, possibly a small figurine or a decorative piece. To the right of the monitor, there's a white mouse pad, a white mouse, and a white keyboard. The mouse pad is positioned in
```

**Task 2 (images/02.jpg):** What is happening in this urban scene? Describe it....

```text
A window with a view of the outside.
```

**Task 3 (images/03.jpg):** Describe the subject and setting of this image....

```text
The image depicts a mountainous landscape with a clear sky. The mountains are primarily composed of dark, shadowed peaks and slopes, indicating a high elevation. The sky is predominantly light blue with a gradient of lighter blue towards the horizon, suggesting a clear day with minimal cloud cover. The mountains are not sharply defined, implying that they are not very tall or steep. The mountains are densely packed, with a few peaks appearing slightly more prominent than others, creating a sense of depth and perspective. The overall scene is serene and tranquil, with the mountains providing a 
```

**Task 4 (images/04.jpg):** What objects and colors are visible in this image?...

```text
The image depicts a landscape scene with a prominent focus on a road and its surroundings. The road is a winding path that appears to be in a rural or semi-rural area, given the natural and open environment. The road is bordered by a grassy area on one side and a rocky embankment on the other. The embankment is composed of a mix of dirt and rocks, with patches of grass and some vegetation visible. The road itself is a dirt path, with a few patches of snow or frost on the surface, indicating that it has recently been cleared or has been recently used. The road is surrounded by
```

**Task 5 (images/05.jpg):** Describe the scenery and atmosphere in this image....

```text
The image depicts a serene lakeside scene at dusk. The main focus is a wooden dock extending into the calm waters of a lake. The dock is constructed from wooden planks and features two metal railings on either side, which are securely attached to the dock. These railings are designed to provide a safe and stable platform for boats or other watercraft to dock. The water around the dock is calm, with gentle ripples indicating the presence of small waves. The sky above is a gradient of soft blue, transitioning from a lighter shade near the horizon to a deeper blue towards the top. This color pale
```

**Task 6 (images/06.jpg):** What does this image show? Give a detailed description....

```text
The image depicts a person wearing a colorful, patterned sweater. The sweater is predominantly composed of multicolored hues, with the main colors being pink, blue, green, and yellow. The sweater has a fringe design at the bottom, adding a touch of elegance to the outfit. The person is standing against a backdrop of a wooden table, which is positioned relatively close to the camera. The table appears to be made of light-colored wood, and it is positioned in the foreground of the image. The person's attire is not the main focus of the image, but it is noticeable due to its vibrant colors and th
```

**Task 7 (images/07.jpg):** Describe the composition and content of this image....

```text
The image depicts a narrow canal in a European city, likely Amsterdam, viewed from a high vantage point. The canal is bordered by buildings on both sides, with a bridge spanning the width of the canal. The buildings are constructed from brick and have a traditional architectural style, with multiple stories and ornate details. The sky above is clear, suggesting a bright and sunny day. The water in the canal is calm, reflecting the light from the sky, giving it a serene and tranquil appearance. A small boat is visible on the water, moving slowly towards the right side of the image. The boat is 
```

**Task 8 (images/08.jpg):** What is the main subject, and what surrounds it?...

```text
The main subject is a bunch of dark purple grapes, and it is surrounded by a person's hands.
```

**Task 9 (images/09.jpg):** Describe what this image depicts in full detail....

```text
The image depicts a dense, snow-covered forest under a foggy sky. The trees are tall and densely packed, creating a dense, almost impenetrable canopy of snow. The snow appears to be fresh, with no signs of melting or melting, indicating that the winter has not yet passed. The trees are covered in a thick layer of snow, with some branches and leaves still visible, suggesting that the season is still winter. The foggy atmosphere obscures the details of the trees and the ground, making it difficult to discern specific features or textures. The forest appears to be located in a mountainous or high
```

**Task 10 (images/10.png):** Read all the text visible in this image and transcribe it ex...

```text
STOP CAFE OPEN Welcome to the New World of AGI. speed matters = micro = 360M.
```

