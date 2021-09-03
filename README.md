# MBART_50_STREAMLIT_TRANSLATOR
- AN OCR AND MULTILINGUAL MACHINE TRANSLATOR - used for extracting text and translating it to required languages.
- Upload a high quality image with proper font and get the required TRANSLATIONS.
### REQUIREMENTS:

* python 3.7+
* pytorch == 1.9.0
* streamlit == 0.86.0

### CREATING VENV IN CONDA:
```sh
conda create --name myenv
```
Using a virtual environment:
```sh
conda activate myenv
```
Deactivating a virtual environment:
```sh
conda deactivate myenv
```
### INSTALLING DEPENDENCIES:
PYTORCH FOR CUDA 10.2:
```sh
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```
PYTORCH - CPU ONLY VERSION:
```sh
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
STREAMLIT:
```sh
pip install streamlit == 0.86.0
```

### RUNNING THE APPLICATION:
run :
```sh
streamlit run app.py
```
- Jump to http://localhost:8501 
- Your app will be hosted in the local server.
- Then deploy the app to cloud platforms but in our case, the free versions of Heroku and Streamlit Sharing won't handle whole 2.4GB pre-trained model (MBART-50).
- The home and the output pages of the web app is shown below:
<div class="row">
    <img src="DEFAULT.png" title='HomePage' alt="index" style="width:30%">
</div>
<div class="row">
    <img src="OUTPUT1.png" title='TEST1' alt="Translated" style="width:30%">
</div>
<div class="row">
    <img src="OUTPUT2.png" title='TEST2' alt="Translated" style="width:30%">
</div>
