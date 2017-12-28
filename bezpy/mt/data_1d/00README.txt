The files in this directory comprise version 1.3 of the USGS' ground 
conductivity model, and are based on:

Fernberg, P. (2012), One-Dimensional Earth Resistivity Models for Selected 
  Areas of Continental United States and Alaska", Electrical Power Research
  Institute (EPRI) Technical Update 1026430, Palo Alto, CA.

Love, J, and Blum, C. (2014 - in prep.), , USGS Open File Report (OFR).

The changes wrt version 1.2 are trivial. The thickness of layer 2 was
corrected from 6km to 5km, and a mis-transcription of the lowest layer
conductivity was corrected to 1.688 S/m. Neither change is expected to
alter calculated impedances significantly.

The .txt files here are intended to be machine readable, although they are 
formatted such that a human can read them with little trouble. In some cases, 
very obvious mistakes/typos were noted in the above references, and corrected 
in these files. It should also be noted that, in some cases, alternative values
were provided for specific regions/depths. We chose what we judged to be the 
most representative of these values to avoid any ambiguity with automated data
readers; this is an admittedly subjective criterion.

Each .txt file is named for, and contains metadata that associate it with, a 
particular conductivity region, for which the data is supposed to be 
representative of the 1D "layer-cake" conductivity of the entire region. These 
regions correspond roughly to long-established physiographic "provinces", but 
can encompass several, or even split one into many. These regions should be 
defined by a Keyhole Markup Language (KML or KMZ) file that can be downloaded
from the USGS website.

Finally, hese conductivity profiles are based on literature reviews, NOT on any
single field study or campaign. As such, they should not be considered 
"definitive", or even self-consistent.

-EJR 10/2014
