import RPi.GPIO as GPIO
from time import sleep

#amount of extension
FULL = 2.5
HALF = 7.5
NONE = 12.5

GPIO.setmode(GPIO.BOARD)

GPIO.setup(3,GPIO.OUT)
GPIO.setup(5,GPIO.OUT)
GPIO.setup(7,GPIO.OUT)
GPIO.setup(11,GPIO.OUT)
GPIO.setup(13,GPIO.OUT)

thumb = GPIO.PWM(3,50)
index = GPIO.PWM(5,50)
middle = GPIO.PWM(7,50)
ring = GPIO.PWM(11,50)
pinky = GPIO.PWM(13,50)

thumb.start(0)
index.start(0)
middle.start(0)
ring.start(0)
pinky.start(0)

def extend(t, i, m, r, p):
  t = (t - 7.5) * -1 + 7.5

  thumb.ChangeDutyCycle(t)
  index.ChangeDutyCycle(i)
  middle.ChangeDutyCycle(m)
  ring.ChangeDutyCycle(r)
  pinky.ChangeDutyCycle(p)

def show_letter(letter):
  match letter:
    case 'a':
      extend(FULL, NONE, NONE, NONE, NONE)
    case 'b':
      extend(NONE, FULL, FULL, FULL, FULL)

show_letter('a')
sleep(1)
show_letter('b')
sleep(1)

print("Done!")