[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5146871.svg)](https://zenodo.org/record/5146871) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6747921.svg)](https://zenodo.org/record/6747921)
## RHAMNOSE 1 condensates in heterologous systems 
This repository contains the analysis scripts associated with the manuscript titled *Arabidopsis thaliana* RHAMNOSE 1 *condensate formation drives UDP-rhamnose synthesis*<sup id="a1">[1](#f1)</sup>. Rhamnose is an essential component of the plant cell wall :seedling: and is synthesized from uridine diphosphate (UDP)-glucose by the RHAMNOSE1 (RHM1) enzyme. RHM1 localizes to biomolecular condensates in plants, but their identity, formation, and function remain elusive. The datasets analyzed here demonstrate that RHAMNOSE 1 does not co-localize with stress granules when transiently expressed in an heterologous system, mammalian cells. This finding was consistent when using a GFP or eptiope tag (FLAG tag), and the finding is recapitulated in yeast and plants in the manuscript. 

## Data management
Images were acquired on a Zeiss LSM900 and all micrographs (.czi) are available in zipped folders with the relevant title and the suffix '_raw-images' from the open-access [Zenodo dataset](https://zenodo.org/record/5146871). For each analysis set, folders of mask images are available in addition to *proofs* which overlay the masked features and the raw image. Please see [analyses](#f20) for analysis details of each dataset, and [reproducing workflow](#f21) for advice if you would like to reproduce the analysis pipeline on your machine.

## Fluorescence image analyses <b id="f20"></b>
### Purpose
Determine whether RHAMNOSE 1 co-occurs with stress granules in an heterologous system. RHAMNOSE 1 tagged with either FLAG or GFP was transiently expressed in U2OS cells and cells were subjecte to incubation with PBS (control), sodium arsenite (stress), or heat shock (stress). Cells were then fixed and probed with an anti-GBP1 antibody to identify stress granules. Images were acquired with a __x objective on a confocal microscope. For lineplots, czi files were opened in FIJI<sup id="a9">[9](#f9)</sup>, line drawn on the RHAMNOSE 1 channel and copied onto the stress granule channel, then pixel intensity values were measured and copied to an excel file for later visualization with matplotlib.

### Data produced
From czi files, cells and nuclei were segmented using CellPose. RHAMNOSE 1 foci were segmented using ___ standard deviations per cell. The mask arrays were export as a pandas dataframe/csv for further processing.

### Figures in manuscript
Used to generate data depicted in ___.

### Analysis software
FIJI<sup id="a9">[9](#f9)</sup> and the following python packages: [CellPose](https://www.cellpose.org/)<sup id="a5">[5](#f5)</sup>, [napari](https://napari.org/)<sup id="a6">[6](#f6)</sup>, [scikit-image](https://scikit-image.org/)<sup id="a8">[8](#f8)</sup>

## Reproducing workflow <b id="f21"></b>
### Prerequisites
Packages required for Python scripts can be accessed in the ```environment.yml``` file available in each analysis folder. To create a new conda environment containing all packages, run ```conda create -f environment.yml```. 

### Workflow
To view analysis results, including masks and validated object classification, all processing steps available as an open-access [Zenodo dataset](https://zenodo.org/record/5146871). To reproduce analysis presented in the manuscript run the ```0_data_retrieval.py``` script for the analysis workflow of interest. The data retrieval script downloads and unzips the original images along with their masks and summary tables. Analysis for the paper was conducted by running the scripts in the enumerated order. To regenerate these results yourself, run the code in the order indicated by the script number for each folder.

## References

<b id="f1">1.</b> Sterling Field, Yanniv Dorone, Will P. Dwyer, Jack A. Cox, Renee Hastings, Madison Blea, Olivia M. S. Carmo, Dan Raba, John Froehlich, Ian S. Wallace, Steven Boeynaems, Seung Y. Rhee. Arabidopsis thaliana RHAMNOSE 1 condensate formation drives UDP-rhamnose synthesis. [bioRxiv 2024.02.15.580454](https://www.biorxiv.org/content/10.1101/2024.02.15.580454v1)[↩](#a1)

<b id="f5">5.</b> Stringer C, Wang T, Michaelos M, Pachitariu M. Cellpose: a generalist algorithm for cellular segmentation. Nat Methods. 2021;18: 100–106. doi:10.1038/s41592-020-01018-x. [↩](#a5)

<b id="f6">6.</b> Sofroniew N, Lambert T, Evans K, Nunez-Iglesias J, Winston P, Bokota G, et al. napari/napari: 0.4.9rc2. Zenodo; 2021. doi:10.5281/zenodo.4915656. [↩](#a6)
updates: napari contributors (2019). napari: a multi-dimensional image viewer for python. doi:10.5281/zenodo.3555620 [↩](#a6)

<b id="f8">8.</b> Walt S van der, Schönberger JL, Nunez-Iglesias J, Boulogne F, Warner JD, Yager N, et al. scikit-image: image processing in Python. PeerJ. 2014;2: e453. doi:10.7717/peerj.453. [↩](#a8)

<b id="f9">9.</b> Schindelin J, Arganda-Carreras I, Frise E, Kaynig V, Longair M, Pietzsch T, et al. Fiji: an open-source platform for biological-image analysis. Nat Methods. 2012;9: 676–682. doi:10.1038/nmeth.2019. [↩](#a9)
