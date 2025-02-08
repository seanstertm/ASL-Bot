import RPi.GPIO as GPIO
from time import sleep

#amount of extension
FULL = 2.5
HALF = 9.5
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
  if t == FULL:
    t = NONE
  elif t == NONE:
    t = FULL
  elif t == HALF:
    t = 7.5

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
    case 'c':
      extend(NONE, HALF, HALF, HALF, HALF)
    case 'd':
      extend(NONE, FULL, NONE, NONE, NONE)
    case 'e':
      extend(HALF, HALF, HALF, HALF, HALF)
    case 'f':
      extend(NONE, NONE, FULL, FULL, FULL)
    case 'g':
      extend(HALF, FULL, NONE, NONE, NONE)
    case 'h':
      extend(NONE, FULL, FULL, NONE, NONE)
    case 'i':
      extend(NONE, NONE, NONE, NONE, FULL)
    case 'j': # This will rotate. Special case
      extend(NONE, NONE, NONE, NONE, FULL)
    case 'k':
      extend(FULL, FULL, FULL, NONE, NONE)
    case 'l':
      extend(FULL, FULL, NONE, NONE, NONE)
    case 'm':
      extend(HALF, HALF, HALF, HALF, NONE)
    case 'n':
      extend(HALF, HALF, HALF, NONE, NONE)
    case 'o':
      extend(NONE, NONE, NONE, NONE, NONE)
    case 'p':
      extend(HALF, FULL, HALF, NONE, NONE)
    case 'q':
      extend(HALF, FULL, NONE, NONE, NONE)
    case 'r':
      extend(NONE, FULL, HALF, NONE, NONE)
    case 's':
      extend(HALF, NONE, NONE, NONE, NONE)
    case 't':
      extend(HALF, HALF, NONE, NONE, NONE)
    case 'u':
      extend(NONE, FULL, FULL, NONE, NONE)
    case 'v':
      extend(NONE, FULL, NONE, FULL, NONE)
    case 'w':
      extend(NONE, FULL, FULL, FULL, NONE)
    case 'x':
      extend(NONE, HALF, NONE, NONE, NONE)
    case 'y':
      extend(FULL, NONE, NONE, NONE, FULL)
    case 'z':
      extend(NONE, FULL, NONE, NONE, NONE)

for letter in "abcdefghijklmnopqrstuvwxyz":
  show_letter(letter)
  sleep(1)

print("Done!")