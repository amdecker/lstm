# lstm
Java Semester 1 Final Project

This is a java implementation of an lstm.

I trained it (one layer of lstm) on the Senate Intelligence Comittee's report on CIA torture that was released in 2014. It can be found here: https://archive.org/stream/SenateCIATortureReport/SenateCIATortureReport_djvu.txt and is also in the project in a file called senate.txt. 

Due to time and computing power restraints, the net has not acutally been trained for a full pass through the training data, but it still works better than I expected (I had very low expectations). The extent of its capabailities are completing words/phrases like "enhanced interrogation tech", "UNCLAS", and "TOP SECR" by adding "niques", "SIFIED", and "ET" resepectively.

To run it you will need to download this file which is too large to put on github and put it into src/main/java: https://drive.google.com/open?id=1OSqIUclU--uMXmItMCpX4l6Y36bpq98F 
