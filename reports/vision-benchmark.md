# Vision-Benchmark Report (all candidates)

Consolidated comparison of every vision candidate benchmarked on the same
10-image task set with llama.cpp (`llama-mtmd-cli`, build from source).

| Candidate | Model file | Weights | Avg tok/s | Wall (10 img) | Tasks |
|---|---|---:|---:|---:|---:|
| gemma3-4b | gemma-3-4b-it-Q4_K_M.gguf | 2.49 GB | None | 22.68s | 10 |
| moondream2 | moondream2-050824-q8.gguf | 1.51 GB | None | 10.06s | 10 |
| smolvlm-2b | SmolVLM-Instruct-Q4_K_M.gguf | 1.11 GB | None | 11.17s | 10 |
| smolvlm-500m | SmolVLM-500M-Instruct-Q8_0.gguf | 0.44 GB | None | 3.81s | 10 |
| smolvlm2-500m | SmolVLM2-500M-Video-Instruct-Q8_0.gguf | 0.44 GB | None | 3.01s | 10 |

## gemma3-4b

- Source: `ggml-org/gemma-3-4b-it-GGUF`
- Files: `gemma-3-4b-it-Q4_K_M.gguf` (2.49 GB) + `mmproj-model-f16.gguf`
- System: x86_64, 4 cores, 16.77 GB RAM
- Wall (10 tasks): 22.68s | avg throughput: None tok/s

| # | Image | Out tok | eval (ms) | tok/s | peak RSS (MB) | wall (s) |
|---|---|---:|---:|---:|---:|---:|
| 1 | images/01.jpg | None | None | None | 5094.5 | 2.338 |
| 2 | images/02.jpg | None | None | None | 5094.77 | 2.208 |
| 3 | images/03.jpg | None | None | None | 5094.65 | 2.26 |
| 4 | images/04.jpg | None | None | None | 5094.58 | 2.21 |
| 5 | images/05.jpg | None | None | None | 5094.59 | 2.286 |
| 6 | images/06.jpg | None | None | None | 5094.66 | 2.296 |
| 7 | images/07.jpg | None | None | None | 5094.51 | 2.271 |
| 8 | images/08.jpg | None | None | None | 5094.64 | 2.285 |
| 9 | images/09.jpg | None | None | None | 5094.56 | 2.259 |
| 10 | images/10.png | None | None | None | 5094.74 | 2.27 |

### Responses

**Task 1 (images/01.jpg):** Describe what you see in this image in detail....

```text

```

**Task 2 (images/02.jpg):** What is happening in this urban scene? Describe it....

```text

```

**Task 3 (images/03.jpg):** Describe the subject and setting of this image....

```text

```

**Task 4 (images/04.jpg):** What objects and colors are visible in this image?...

```text

```

**Task 5 (images/05.jpg):** Describe the scenery and atmosphere in this image....

```text

```

**Task 6 (images/06.jpg):** What does this image show? Give a detailed description....

```text

```

**Task 7 (images/07.jpg):** Describe the composition and content of this image....

```text

```

**Task 8 (images/08.jpg):** What is the main subject, and what surrounds it?...

```text

```

**Task 9 (images/09.jpg):** Describe what this image depicts in full detail....

```text

```

**Task 10 (images/10.png):** Read all the text visible in this image and transcribe it ex...

```text

```


---

## moondream2

- Source: `cjpais/moondream2-llamafile`
- Files: `moondream2-050824-q8.gguf` (1.51 GB) + `moondream2-mmproj-050824-f16.gguf`
- System: x86_64, 4 cores, 16.77 GB RAM
- Wall (10 tasks): 10.06s | avg throughput: None tok/s

| # | Image | Out tok | eval (ms) | tok/s | peak RSS (MB) | wall (s) |
|---|---|---:|---:|---:|---:|---:|
| 1 | images/01.jpg | None | None | None | 5392.47 | 1.089 |
| 2 | images/02.jpg | None | None | None | 5394.02 | 0.998 |
| 3 | images/03.jpg | None | None | None | 5392.54 | 0.992 |
| 4 | images/04.jpg | None | None | None | 5392.45 | 0.985 |
| 5 | images/05.jpg | None | None | None | 5392.57 | 0.99 |
| 6 | images/06.jpg | None | None | None | 5392.5 | 1.007 |
| 7 | images/07.jpg | None | None | None | 5392.5 | 1.0 |
| 8 | images/08.jpg | None | None | None | 5392.51 | 1.014 |
| 9 | images/09.jpg | None | None | None | 5392.51 | 1.0 |
| 10 | images/10.png | None | None | None | 5392.26 | 0.99 |

### Responses

**Task 1 (images/01.jpg):** Describe what you see in this image in detail....

```text

```

**Task 2 (images/02.jpg):** What is happening in this urban scene? Describe it....

```text

```

**Task 3 (images/03.jpg):** Describe the subject and setting of this image....

```text

```

**Task 4 (images/04.jpg):** What objects and colors are visible in this image?...

```text

```

**Task 5 (images/05.jpg):** Describe the scenery and atmosphere in this image....

```text

```

**Task 6 (images/06.jpg):** What does this image show? Give a detailed description....

```text

```

**Task 7 (images/07.jpg):** Describe the composition and content of this image....

```text

```

**Task 8 (images/08.jpg):** What is the main subject, and what surrounds it?...

```text

```

**Task 9 (images/09.jpg):** Describe what this image depicts in full detail....

```text

```

**Task 10 (images/10.png):** Read all the text visible in this image and transcribe it ex...

```text

```


---

## smolvlm-2b

- Source: `ggml-org/SmolVLM-Instruct-GGUF`
- Files: `SmolVLM-Instruct-Q4_K_M.gguf` (1.11 GB) + `mmproj-SmolVLM-Instruct-f16.gguf`
- System: x86_64, 4 cores, 16.77 GB RAM
- Wall (10 tasks): 11.17s | avg throughput: None tok/s

| # | Image | Out tok | eval (ms) | tok/s | peak RSS (MB) | wall (s) |
|---|---|---:|---:|---:|---:|---:|
| 1 | images/01.jpg | None | None | None | 4210.5 | 1.276 |
| 2 | images/02.jpg | None | None | None | 4210.4 | 1.099 |
| 3 | images/03.jpg | None | None | None | 4210.67 | 1.1 |
| 4 | images/04.jpg | None | None | None | 4210.54 | 1.102 |
| 5 | images/05.jpg | None | None | None | 4210.56 | 1.1 |
| 6 | images/06.jpg | None | None | None | 4210.54 | 1.1 |
| 7 | images/07.jpg | None | None | None | 4210.55 | 1.101 |
| 8 | images/08.jpg | None | None | None | 4210.54 | 1.101 |
| 9 | images/09.jpg | None | None | None | 4210.63 | 1.099 |
| 10 | images/10.png | None | None | None | 4210.58 | 1.097 |

### Responses

**Task 1 (images/01.jpg):** Describe what you see in this image in detail....

```text

```

**Task 2 (images/02.jpg):** What is happening in this urban scene? Describe it....

```text

```

**Task 3 (images/03.jpg):** Describe the subject and setting of this image....

```text

```

**Task 4 (images/04.jpg):** What objects and colors are visible in this image?...

```text

```

**Task 5 (images/05.jpg):** Describe the scenery and atmosphere in this image....

```text

```

**Task 6 (images/06.jpg):** What does this image show? Give a detailed description....

```text

```

**Task 7 (images/07.jpg):** Describe the composition and content of this image....

```text

```

**Task 8 (images/08.jpg):** What is the main subject, and what surrounds it?...

```text

```

**Task 9 (images/09.jpg):** Describe what this image depicts in full detail....

```text

```

**Task 10 (images/10.png):** Read all the text visible in this image and transcribe it ex...

```text

```


---

## smolvlm-500m

- Source: `ggml-org/SmolVLM-500M-Instruct-GGUF`
- Files: `SmolVLM-500M-Instruct-Q8_0.gguf` (0.44 GB) + `mmproj-SmolVLM-500M-Instruct-f16.gguf`
- System: x86_64, 4 cores, 16.77 GB RAM
- Wall (10 tasks): 3.81s | avg throughput: None tok/s

| # | Image | Out tok | eval (ms) | tok/s | peak RSS (MB) | wall (s) |
|---|---|---:|---:|---:|---:|---:|
| 1 | images/01.jpg | None | None | None | 1344.0 | 0.383 |
| 2 | images/02.jpg | None | None | None | 1343.8 | 0.381 |
| 3 | images/03.jpg | None | None | None | 1343.67 | 0.382 |
| 4 | images/04.jpg | None | None | None | 1343.65 | 0.384 |
| 5 | images/05.jpg | None | None | None | 1343.63 | 0.38 |
| 6 | images/06.jpg | None | None | None | 1343.83 | 0.378 |
| 7 | images/07.jpg | None | None | None | 1343.63 | 0.377 |
| 8 | images/08.jpg | None | None | None | 1343.86 | 0.38 |
| 9 | images/09.jpg | None | None | None | 1343.75 | 0.384 |
| 10 | images/10.png | None | None | None | 1343.48 | 0.38 |

### Responses

**Task 1 (images/01.jpg):** Describe what you see in this image in detail....

```text

```

**Task 2 (images/02.jpg):** What is happening in this urban scene? Describe it....

```text

```

**Task 3 (images/03.jpg):** Describe the subject and setting of this image....

```text

```

**Task 4 (images/04.jpg):** What objects and colors are visible in this image?...

```text

```

**Task 5 (images/05.jpg):** Describe the scenery and atmosphere in this image....

```text

```

**Task 6 (images/06.jpg):** What does this image show? Give a detailed description....

```text

```

**Task 7 (images/07.jpg):** Describe the composition and content of this image....

```text

```

**Task 8 (images/08.jpg):** What is the main subject, and what surrounds it?...

```text

```

**Task 9 (images/09.jpg):** Describe what this image depicts in full detail....

```text

```

**Task 10 (images/10.png):** Read all the text visible in this image and transcribe it ex...

```text

```


---

## smolvlm2-500m

- Source: `ggml-org/SmolVLM2-500M-Video-Instruct-GGUF`
- Files: `SmolVLM2-500M-Video-Instruct-Q8_0.gguf` (0.44 GB) + `mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf`
- System: x86_64, 4 cores, 16.77 GB RAM
- Wall (10 tasks): 3.01s | avg throughput: None tok/s

| # | Image | Out tok | eval (ms) | tok/s | peak RSS (MB) | wall (s) |
|---|---|---:|---:|---:|---:|---:|
| 1 | images/01.jpg | None | None | None | 988.07 | 0.439 |
| 2 | images/02.jpg | None | None | None | 988.0 | 0.285 |
| 3 | images/03.jpg | None | None | None | 988.01 | 0.284 |
| 4 | images/04.jpg | None | None | None | 988.1 | 0.283 |
| 5 | images/05.jpg | None | None | None | 988.09 | 0.283 |
| 6 | images/06.jpg | None | None | None | 988.07 | 0.294 |
| 7 | images/07.jpg | None | None | None | 988.11 | 0.285 |
| 8 | images/08.jpg | None | None | None | 988.12 | 0.286 |
| 9 | images/09.jpg | None | None | None | 988.09 | 0.285 |
| 10 | images/10.png | None | None | None | 988.06 | 0.285 |

### Responses

**Task 1 (images/01.jpg):** Describe what you see in this image in detail....

```text

```

**Task 2 (images/02.jpg):** What is happening in this urban scene? Describe it....

```text

```

**Task 3 (images/03.jpg):** Describe the subject and setting of this image....

```text

```

**Task 4 (images/04.jpg):** What objects and colors are visible in this image?...

```text

```

**Task 5 (images/05.jpg):** Describe the scenery and atmosphere in this image....

```text

```

**Task 6 (images/06.jpg):** What does this image show? Give a detailed description....

```text

```

**Task 7 (images/07.jpg):** Describe the composition and content of this image....

```text

```

**Task 8 (images/08.jpg):** What is the main subject, and what surrounds it?...

```text

```

**Task 9 (images/09.jpg):** Describe what this image depicts in full detail....

```text

```

**Task 10 (images/10.png):** Read all the text visible in this image and transcribe it ex...

```text

```

