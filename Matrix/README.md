# Test matrices

| Name       |         n |  # nonzero |
| :--------- | --------: | ---------: |
| atmosmodd  | 1,270,432 |  8,814,880 |
| atmosmodm  | 1,489,752 | 10,319,760 |
| G3_circuit | 1,585,478 |  7,660,826 |
| thermal2   | 1,228,045 |  8,580,313 |
| thermal1   |    82,654 |    574,458 |
| Transport  | 1,602,111 | 23,487,281 |
| t2em       |   921,632 |  4,590,832 |
| wang3      |    26,064 |    177,168 |
| wang4      |    26,068 |    177,196 |

## Downloading the matrices

The experiments use the above matrices selected from [the SuiteSparse Collection Matrix](https://suitesparse-collection-website.herokuapp.com/), a matrix database. However, we do not contain them in this repository due to the file size. Thus first, they must be downloaded in this directory in some way.

We provide a [download.sh](download.sh) script to do that. You can download all matrices by running the script as follows. 

```shell
cd (path to this directory)
zsh download.sh
```

Note that you also can download them from [the SuiteSparse Collection Matrix](https://suitesparse-collection-website.herokuapp.com) manually.

