# AI-Motion

<img src="assets/demo.gif" width="600" alt="AI-Motion Demo">

*Live-Erkennung der sieben Emotionen über die Webcam*

## Inhaltsverzeichnis

- [Einführung](#einführung)
- [Voraussetzungen und Abhängigkeiten](#voraussetzungen-und-abhängigkeiten)
- [Grundlegende Nutzung](#grundlegende-nutzung)
  - [Schritt 1: Projekt von Git klonen](#schritt-1-projekt-von-git-klonen)
  - [Schritt 2: Virtuelle Umgebung einrichten](#schritt-2-virtuelle-umgebung-einrichten)
  - [Schritt 3: Libraries installieren](#schritt-3-libraries-installieren)
  - [Schritt 4: Programm ausführen](#schritt-4-programm-ausführen)
  - [Ordnerstruktur](#ordnerstruktur)
- [Algorithmus](#algorithmus)
- [Erstellung einer ausführbaren Datei (.exe)](#erstellung-einer-ausführbaren-datei-exe)
  - [Anforderungen](#anforderungen)
  - [Schritt 1: PyInstaller installieren](#schritt-1-pyinstaller-installieren)
  - [Schritt 2: Projektverzeichnis auswählen](#schritt-2-projektverzeichnis-auswählen)
  - [Schritt 3: Skript in eine ausführbare Datei umwandeln](#schritt-3-skript-in-eine-ausführbare-datei-umwandeln)
  - [Schritt 4: Dateien umstrukturieren](#schritt-4-dateien-umstrukturieren)
  - [Schritt 5: Batch-Datei erstellen](#schritt-5-batch-datei-erstellen)
- [Fehlerbehebung](#fehlerbehebung)

## Einführung

Dieses Projekt zielt darauf ab, die Emotion auf dem Gesicht einer Person in eine von **sieben Kategorien** einzuordnen, indem tiefe konvolutionelle neuronale Netze verwendet werden. Das Modell wird auf dem **FER-2013** Datensatz trainiert, der auf der International Conference on Machine Learning (ICML) veröffentlicht wurde. Dieser Datensatz besteht aus 35.887 Graustufen-Bildern von Gesichtern in der Größe 48x48 Pixel mit **sieben Emotionen**: wütend, angewidert, ängstlich, glücklich, neutral, traurig und überrascht.

## Voraussetzungen und Abhängigkeiten

Für die Verwendung des Projekts werden folgende Voraussetzungen benötigt:

- Windows
- Python 3.11
- Git
- Webcam
- pip (Python-Paket-Manager)

Für das Projekt werden folgende Python-Bibliotheken verwendet:

- NumPy 2.0.2
- argparse 1.4.0
- Matplotlib 3.9.3
- OpenCV 4.10.0.84
- TensorFlow 2.18.0

> **Hinweis:** Für dieses Projekt wird Python 3.11 verwendet, da die verwendete TensorFlow-Version mit dieser Python-Version kompatibel ist. Eine andere Python-Version kann zu Problemen bei der Installation oder Ausführung führen.

## Grundlegende Nutzung

### Schritt 1: Projekt von Git klonen

Öffnen Sie PowerShell und navigieren Sie zunächst zu dem Ordner, in dem das Projekt gespeichert werden soll.

Klonen Sie anschließend das Repository:

```bash
git clone https://github.com/maxlindnerhtl/aimotion.git
cd aimotion
```

Dadurch wird das Projekt heruntergeladen und ein neuer Ordner `aimotion` erstellt.

### Schritt 2: Virtuelle Umgebung einrichten

Für das Projekt wird eine eigene virtuelle Python-Umgebung verwendet. Dadurch bleiben die Abhängigkeiten des Projekts von der globalen Python-Installation getrennt.

Erstellen Sie die virtuelle Umgebung mit Python 3.11:

```bash
py -3.11 -m venv .venv
```

Aktivieren Sie anschließend die virtuelle Umgebung:

```bash
.\.venv\Scripts\Activate.ps1
```

Nach der Aktivierung sollte `(.venv)` am Anfang der Kommandozeile angezeigt werden.

> **Hinweis:** Wenn `py -3.11` nicht funktioniert, ist Python 3.11 möglicherweise nicht installiert oder nicht über den Python Launcher verfügbar.

### Schritt 3: Libraries installieren

Stellen Sie sicher, dass die virtuelle Umgebung weiterhin aktiviert ist. Anschließend können die benötigten Libraries mithilfe von `requirements.txt` installiert werden:

```bash
pip install -r src\requirements.txt
```

Nach erfolgreicher Installation sollten keine Fehlermeldungen angezeigt werden.

### Schritt 4: Programm ausführen

Navigieren Sie in den Quellordner:

```bash
cd src
```

Starten Sie anschließend das Programm:

```bash
python emotions.py --mode display --overlay overlay.png
```

Nach dem Start wird die Webcam geöffnet. Erkannte Gesichter werden analysiert und die vorhergesagte Emotion wird im Kamerabild angezeigt.

Das Programm lässt sich durch Betätigen der Taste `Q` beenden. Alternativ kann das Programm durch Schließen des Fensters beendet werden.

### Ordnerstruktur

Nach dem Klonen des Projekts sieht die grundlegende Struktur wie folgt aus:
```
aimotion/
├── README.md
└── src/
├── data/
├── emotions.py
├── haarcascade_frontalface_default.xml
├── overlay.png
├── model.h5
└── requirements.txt
```

Dabei befinden sich die für die Anwendung benötigten Python-Dateien, das trainierte Modell, die Haar-Cascade und das Overlay-Bild im Ordner `src`.

## Algorithmus

1. **Gesichtserkennung:** Die **Haar-Cascade**-Methode wird verwendet, um Gesichter in jedem Frame des Webcam-Feeds zu erkennen.
2. **Bildvorbereitung:** Der Bereich des Bildes, der das Gesicht enthält, wird auf **48x48 Pixel** skaliert und für die Verarbeitung durch das neuronale Netzwerk vorbereitet.
3. **Emotionserkennung:** Das **CNN (Convolutional Neural Network)** analysiert das vorbereitete Gesicht und gibt eine Liste von **Softmax-Scores** für die sieben Emotionen aus. Die Scores geben an, wie wahrscheinlich das Modell die jeweilige Emotion einschätzt.
4. **Auswahl der Emotion:** Die Emotion mit dem höchsten Score wird als Vorhersage ausgewählt.
5. **Anzeige:** Die erkannte Emotion wird anschließend auf dem Bildschirm angezeigt.

---

## Erstellung einer ausführbaren Datei (.exe)

Dieser Abschnitt ist optional.

Für die normale Verwendung des Projekts wird keine .exe-Datei benötigt. PyInstaller wird nur verwendet, wenn das Python-Programm als ausführbare Windows-Datei gestartet werden soll, ohne den Python-Befehl jedes Mal manuell ausführen zu müssen.

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

Außerdem muss das Projekt bereits eingerichtet und die virtuelle Umgebung aktiviert sein.

### Schritt 1: PyInstaller installieren

Installieren Sie PyInstaller mit folgendem Befehl:

```bash
pip install pyinstaller
```

PyInstaller wird verwendet, um das Python-Programm in eine ausführbare .exe-Datei umzuwandeln.

### Schritt 2: Projektverzeichnis auswählen

Navigieren Sie in der Kommandozeile in das Quellverzeichnis Ihres Projekts:

```bash
cd <projektverzeichnis>/src
```

Ersetzen Sie `<projektverzeichnis>` durch den tatsächlichen Pfad zu Ihrem Projekt.

### Schritt 3: Skript in eine ausführbare Datei umwandeln

Führen Sie den folgenden PyInstaller-Befehl aus:

```bash
pyinstaller --onefile --add-data "haarcascade_frontalface_default.xml;." --add-data "overlay.png;." emotions.py
```

Der Parameter `--onefile` sorgt dafür, dass PyInstaller eine einzelne ausführbare .exe-Datei erstellt.

Die zusätzlichen `--add-data`-Parameter sorgen dafür, dass die für das Programm benötigte Haar-Cascade und das Overlay-Bild in die Anwendung eingebunden werden.

Nach erfolgreicher Ausführung erstellt PyInstaller unter anderem einen `build`- und einen `dist`-Ordner.

### Schritt 4: Dateien umstrukturieren

Nach der Ausführung des PyInstaller-Befehls befindet sich die erstellte `emotions.exe` im `dist`-Ordner.

Kopieren Sie die Datei `emotions.exe` aus dem `dist`-Ordner in das übergeordnete Verzeichnis, also in den Ordner `src`.

Anschließend können der `dist`- und der `build`-Ordner gelöscht werden, da sie für die spätere Verwendung der Anwendung nicht mehr benötigt werden.

PyInstaller erstellt außerdem automatisch die Datei `emotions.spec`. Diese Datei enthält die Konfiguration für den Build und kann für spätere Builds wiederverwendet werden.

#### Endgültige Ordnerstruktur
```
aimotion/
├── README.md
└── src/
├── data/
├── emotions.py
├── emotions.exe
├── emotions.spec
├── images.png
├── haarcascade_frontalface_default.xml
├── model.h5
└── requirements.txt
```


### Schritt 5: Batch-Datei erstellen

Eine `.bat`-Datei kann verwendet werden, um das Programm bequem mit einem Doppelklick zu starten.

Erstellen Sie beispielsweise eine Datei mit dem Namen `start.bat` und fügen Sie folgenden Inhalt ein:

```bat
@echo off
cd <projektverzeichnis>/src
emotions.exe --mode display --overlay overlay.png
```

Ersetzen Sie `<projektverzeichnis>` durch den tatsächlichen Pfad in Ihrem System.

Die Batch-Datei übernimmt das Wechseln in das richtige Verzeichnis und startet anschließend die Anwendung. Dadurch muss der Startbefehl nicht jedes Mal manuell in PowerShell eingegeben werden.

Speichern Sie die `.bat`-Datei und starten Sie das Programm anschließend durch einen Doppelklick auf diese Datei.

> Das Programm lässt sich durch Betätigen der Taste `Q` beenden. Alternativ kann das Programm durch Schließen des Fensters beendet werden.

---

## Fehlerbehebung

### `Python was not found` oder `py -3.11` funktioniert nicht

Stellen Sie sicher, dass **Python 3.11** installiert ist und über den Python Launcher verfügbar ist.

Prüfen Sie die installierte Python-Version mit:

```bash
py -3.11 --version
```

### No module named ...

Stellen Sie sicher, dass die virtuelle Umgebung aktiviert ist. In der Kommandozeile sollte `(.venv)` angezeigt werden.

Falls die benötigten Libraries noch nicht installiert wurden, führen Sie erneut folgenden Befehl aus:

```bash
pip install -r src\requirements.txt
```

### Could not load overlay image

Überprüfen Sie, ob die Datei `overlay.png` im Ordner `src` vorhanden ist und der Dateiname im Startbefehl korrekt geschrieben wurde:

```bash
python emotions.py --mode display --overlay overlay.png
```

### Die Webcam öffnet sich nicht

Stellen Sie sicher, dass eine Webcam angeschlossen bzw. verfügbar ist und nicht bereits von einer anderen Anwendung verwendet wird.

Überprüfen Sie außerdem, ob Windows der Anwendung den Zugriff auf die Kamera erlaubt.

### Das Programm lässt sich nicht starten

Stellen Sie sicher, dass Sie sich im richtigen Verzeichnis befinden und die virtuelle Umgebung aktiviert ist.

Für die normale Python-Ausführung sollten Sie sich im Ordner `src` befinden:

```bash
cd src
```

Anschließend kann das Programm mit folgendem Befehl gestartet werden:

```bash
python emotions.py --mode display --overlay overlay.png
```