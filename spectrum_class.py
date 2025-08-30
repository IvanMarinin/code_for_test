from signal_class import Signal


class Spectrum(Signal):
    def __init__(self, filename, wavelengths, intensities,
                 h_u=None, h_g=None, h_e=None, h_p=None,
                 h_s=None, h_m=None, h_i=None):
        """
        Spectrum extends Signal by adding defect-related parameters.
        Each spectrum has optional defect parameters h_u, h_g, h_e, h_p, h_s, h_m, h_i.
        """
        super().__init__(filename, wavelengths, intensities)
        self.h_u = h_u
        self.h_g = h_g
        self.h_e = h_e
        self.h_p = h_p
        self.h_s = h_s
        self.h_m = h_m
        self.h_i = h_i

    @property
    def get_defect_params_list(self):
        """
        Returns defect parameters as a list in a fixed order.
        Useful for building DataFrames or correlation analysis.
        """
        return [
            self.h_i,
            self.h_u,
            self.h_g,
            self.h_e,
            self.h_p,
            self.h_s,
            self.h_m
        ]