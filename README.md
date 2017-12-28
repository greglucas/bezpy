Bezpy
=====

Bezpy is an open source library for analysis of magnetic (B), electric (E),
and impedance (Z) data within a geophysical framework. This library contains
routines for calculating the geoelectric field from the geomagnetic field in
multiple different ways.

##Features
- Geomagnetic to geoelectric field calculations
- Integration of the geoelectric field along transmission lines
- Built using established, fast, open source python libraries
    [Pandas](http://www.pandas.pydata.org/),
    [NumPy](http://www.numpy.org/),
    [SciPy](http://www.scipy.org/)

## Examples
> [More Examples and notebooks in docs/example/](./docs/example/)

The following example reads magnetic field data that in IAGA2002 format
into a Pandas DataFrame.

```import bezpy

df_mag = bezpy.mag.read_iaga("mag_data_2017_BOU.iaga")
df_mag.head()
```

## Install
> [More Install options in docs/install.md](./docs/install.md).

## License
The code is released under a BSD-3 license
[License described in LICENSE.md](./LICENSE.md)


## Problems or Questions?

- [Report an issue using the GitHub issue tracker](http://github.com/usgs/geomag-algorithms/issues)

## Additional Links

- [USGS Geomagnetism Program Home Page](http://geomag.usgs.gov/)
