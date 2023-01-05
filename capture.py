#!/usr/bin/env python
# Benjamin James Wright <bwright@cse.unsw.edu.au>
# This is a simple capture program that will capture a section of
# the screen on a decided interval and then output the images
# with their corresponding timestamps. (Borrowed from stackoverflow).
# This will continue to capture for the seconds specified by argv[1]

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Gdk', '3.0')
from gi.repository import Gdk, GdkPixbuf, Gtk

import time
import sys
import config

print("Starting Capture")
print("================")

time.sleep(config.CAPTURE_STARTUP_DELAY)


root_w = Gdk.get_default_root_window()
w = root_w.get_screen().get_active_window()
x, y, width, height = w.get_geometry()

for i in range(0, config.CAPTURE_NUM):
  root_w.process_all_updates()
  root_w.flush()
  w.invalidate_rect(None, True)
  root_w.invalidate_rect(None, True)

  pb = Gdk.pixbuf_get_from_window(w, 0, 0, width, height)

  if (pb is not None):
    pb.savev(f"capture/screenshot_{time.time()}.png", "png", [], [])
    print(f"Screenshot {i} saved.")
  else:
    print("Unable to get the screenshot.")
  pb.fill(0xFFFFFFFF)
  time.sleep(config.CAPTURE_DELAY)
