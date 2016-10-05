""" General data functions """


def nomodel(ccd_results):
   """Removes models from ccd_results"""
   return ccd_results._replace(mod)
