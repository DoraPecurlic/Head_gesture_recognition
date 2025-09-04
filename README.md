# Head_gesture_recognition
Sustav za upravljanje reprodukcijom videozapisa unutar Youtube playliste koristeći prepoznavanje pokreta glave.
Koristeći kameru i duboko učenje, aplikacija prepoznaje geste glave i izvodi odgovarajuće akcije.

## Struktura projekta
│── models/ 
│ ├── best_val_acc_model.h5  # istrenirani model
│ └── label_encoder.pkl
│
│── src/ 
│ └── Head_gesture_control.py # skripta koja omogućava upravljanje Youtube playlistom pmoću gesti glave
│
│── requirements.txt 
│── README.md 

## Okruženje
  Python 3.11.
  Numpy 
  Tensorflow 
  MediaPipe 
  OpenCV
  PyAutoGUI

  Instalacija svih potrebnih paketa:
  ```
  pip install -r requirements.txt
  ```

## Pokretanje
 Za pokretanje aplikacije:
 python src/head_gesture_control.py


  
