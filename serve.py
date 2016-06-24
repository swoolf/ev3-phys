#!/usr/bin/env python

import webhost as webb

Gsite = webb.webScreen()
Gsite.post("", "index.html")
Gsite.serve()