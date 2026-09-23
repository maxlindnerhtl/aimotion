# AI-Motion

## Einführung

Dieses Projekt zielt darauf ab, die Emotion auf dem Gesicht einer Person in eine von **sieben Kategorien** einzuordnen, indem tiefe konvolutionelle neuronale Netze verwendet werden. Das Modell wird auf dem **FER-2013** Datensatz trainiert, der auf der International Conference on Machine Learning (ICML) veröffentlicht wurde. Dieser Datensatz besteht aus 35.887 Graustufen-Bildern von Gesichtern in der Größe 48x48 Pixel mit **sieben Emotionen**: wütend, angewidert, ängstlich, glücklich, neutral, traurig und überrascht.

## Abhängigkeiten

- Python 3.11
- OpenCV 4.10
- Tensorflow 2.18

## Virtuelle Umgebung einrichten

Für das Projekt wird eine eigene virtuelle Python-Umgebung verwendet. Dadurch bleiben die Abhängigkeiten des Projekts von der globalen Python-Installation getrennt.

Erstellen Sie die virtuelle Umgebung mit Python 3.11:
```bash
py -3.11 -m venv .venv
```
Aktivieren Sie die virtuelle Umgebung:

```bash
.\.venv\Scripts\Activate.ps1
```
Nach der Aktivierung sollte (.venv) am Anfang der Kommandozeile angezeigt werden.

Anschließend können die Libraries mithilfe von requirements.txt installiert werden:
```bash
pip install -r src\requirements.txt
```

## Grundlegende Nutzung

### Schritt 1: Klonen Sie das Projekt von Git:

```bash
https://github.com/maxlindnerhtl/aimotion.git
```

### Schritt 2: Navigieren Sie in den Quellordner und führen Sie das Programm aus:

```bash
cd src
python emotions.py --mode display --overlay overlay.png
```

### Ordnerstruktur

Die Projektstruktur sieht wie folgt aus:

```
src/
  |- data/
  |- emotions.py
  |- haarcascade_frontalface_default.xml
  |- images.png
  |- model.h5
  |- requirements.txt
```

## Algorithmus

1. **Gesichtserkennung:** Die **Haar-Cascade**-Methode wird verwendet, um Gesichter in jedem Frame des Webcam-Feeds zu erkennen.
2. **Bildvorbereitung:** Der Bereich des Bildes, der das Gesicht enthält, wird auf **48x48** Pixel skaliert.
3. **Emotionserkennung:** Das CNN-Modell gibt eine Liste von **Softmax-Scores** für die sieben Klassen von Emotionen aus.
4. **Anzeige:** Die Emotion mit dem höchsten Score wird auf dem Bildschirm angezeigt.

---

## Erstellung einer ausführbaren Datei (.exe)

### Anforderungen

Stellen Sie vor dem Fortfahren sicher, dass folgende Tools und Pakete installiert sind:

- **Python 3.11**
- **pip** (Python-Paket-Manager)
- Python-Bibliotheken:
  - `numpy 2.0.2`
  - `argparse 1.4.0`
  - `matplotlib 3.9.3`
  - `opencv-python 4.10.0.84`
  - `tensorflow 2.18.0`

### Schritt 1: PyInstaller installieren

Installieren Sie PyInstaller mit folgendem Befehl:

```bash
pip install pyinstaller
```

### Schritt 2: Projektverzeichnis auswählen

Navigieren Sie in der Kommandozeile in das Quellverzeichnis Ihres Projekts:

```bash
cd <projektverzeichnis>/src
```

### Schritt 3: Skript in eine ausführbare Datei umwandeln

Führen Sie den folgenden PyInstaller-Befehl aus, um das Skript zu konvertieren:

```bash
pyinstaller --onefile --add-data "haarcascade_frontalface_default.xml;." --add-data "<pfad-zum-overlay-bild>/images.png;src" emotions.py
```

### Schritt 4: Dateien umstrukturieren

Nach der Ausführung des PyInstaller-Befehls sind einige Anpassungen an der Ordnerstruktur erforderlich:

1. Kopieren Sie die benötigten Dateien aus dem `dist`-Ordner in das übergeordnete Verzeichnis, das Quellverzeichnis:
  - `emotions.exe`
2. Löschen Sie den `dist`- und `build`-Ordner, da diese nicht mehr benötigt werden.

#### Endgültige Ordnerstruktur

```
.../src/
  |- data/
  |- emotions.py
  |- emotions.exe
  |- emotions.spec
  |- images.png
  |- haarcascade_frontalface_default.xml
  |- model.h5
  |- requirements.txt
```

### Schritt 5: Batch-Datei erstellen

Erstellen Sie eine `.bat`-Datei, um das Programm bequem auszuführen. Fügen Sie folgenden Inhalt ein:

```bat
rem @echo off
cd <projektverzeichnis>/src
emotions.exe --mode display --overlay images.png
```

Ersetzen Sie `<projektverzeichnis>` durch den tatsächlichen Pfad in Ihrem System.

Speichern Sie die `.bat`-Datei und starten Sie das Programm einfach durch einen Doppelklick auf diese Datei.

Das Programm lässt sich durch betätigen der Taste "Q" beenden. Alternativ lässt es sich auch durch schließen des Skripts beenden

---
