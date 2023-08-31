# ISMulator
Simple model of the thermal structure of the cold interstellar medium

To run the widget, install the streamlit python package and run
``streamlit run ismulator.py``

or visit [https://ismulator.streamlit.app/
](https://ismulator.streamlit.app/)

This was a quick weekend project and should be considered a "toy" model: it uses highly approximate expressions for different cooling and heating processes that *do* capture their key dimensional scalings, but are not generally as accurate as more-sophisticated treatments. If ISMulator experiments give you a cool idea, I recommend following up with tried-and-true research-grade codes like [DESPOTIC](https://bitbucket.org/krumholz/despotic/src/master/) or [Cloudy](https://gitlab.nublado.org/cloudy/cloudy/-/wikis/home) to make sure it works out.
