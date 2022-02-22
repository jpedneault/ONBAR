#-------------------------Import-------------------------------------
import pandas as pd
import pyomo.environ as po
from pyomo.environ import *
import numpy as np

import time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plotly.graph_objects as go


opt = SolverFactory('ipopt')

#----------------------------Class-----------------------------------

class Onbar:
    """
    ONBAR, for «Optimisation of Number of Bins in Aluminium Recycling» is an theoretical optimisation model that minimize
    the environmental impact of the aluminium industry. The optimisation is solve according different number of
    bins and different years. The model is build around two transfer matrices, one for sorting the different
    product into a define number of bins, the second for mixing those bins into different alloys. Finally, sweetening is
    made in order to have the appropriate number alloy composition.

    This code has been used for the calculation of the article in revision:
    Pedneault, J., Majeau-Bettez, G., Margni, M. (2022). How much sorting is required for a circular low carbon
    aluminium economy?

    """

    # Initializer / Instance attributes
    def __init__(self, inflow, outflow, collection_rate, theta, impact_data, ssp, region, b_max, impact, dismantling_condition):
        '''
        Initialise all attributes of Onbar

          Args:
         -----
            inflow: [df] demand flow based on the dMFA
            outflow: [df] flow to recycle based on the dMFA
            collection rate: [Series] Collection rate according different regions and sector
            theta: [df] alloy composition
            impact_data: [df] impacts factor  of different process (production, sorting, remelting).
             
            ssp: [str] ssp of the analysis between: SSP1, SSP2, SSP3, SSP4, SSP5
            region: [str] region of the analysis between: ASIA, LAM, MAF, OECD, REF, GLO
             
            b_max: [int] maximum number of bins to solve the system
             
            impact: [str] Choice of environmental indicator to optimise. Between CC (Cliamte change) and RE (Resource)
            dismantling_condition: [str] Choice of sorting condition. Between 'full' and 'none'
        '''
        # Initial attributes
        self.inflow = inflow.loc[ssp]
        self.outflow = outflow.loc[ssp]
        self.collection_rate = collection_rate.loc[ssp]
        self.theta = (theta.T / theta.sum(1)).T  # Get rid of rounding errors, ensure Theta values add up to one..
        self.impact = impact
        self.ssp = ssp
        self.region = region
        self.b_max = b_max
        self.dismantling_condition = dismantling_condition

        #Impact intensity for this specific class
        name = 'Al-' + self.ssp
        to_drop = ['Al-SSP1', 'Al-SSP2', 'Al-SSP3', 'Al-SSP4', 'Al-SSP5']
        to_drop.remove(name)
        #Get the impact data
        if self.impact == 'CC':
            impact_data_temp = impact_data.drop(to_drop, level=1)
            impact_data_temp.rename(index={name: 'Al'}, inplace=True)
            self.impact_intensity = impact_data_temp.loc[self.impact]
        if self.impact == 'RE':
            self.impact_intensity = impact_data.loc[self.impact]
            
        # Creation of other attributes that will be calculated
        #   Flows used as input of the optimisation model
        self.f_tr = pd.DataFrame()
        self.f_tw = pd.DataFrame()
        self.f_d = pd.DataFrame()
        #   Results of the optimisation model
        self.f_tp_all = pd.DataFrame()
        self.f_r_all = pd.DataFrame()
        self.f_s_all = pd.DataFrame()
        self.f_m_all = pd.DataFrame()
        self.alpha_all = pd.DataFrame()
        self.beta_all = pd.DataFrame()
        self.f_sw_all = pd.DataFrame()
        self.f_p_all = pd.DataFrame()
        self.f_tp_all = pd.DataFrame()
        self.imp_all = pd.DataFrame()
        #   Results of the ideal case
        self.imp_id = pd.DataFrame()
        self.f_tp_id = pd.DataFrame()
        self.f_scrap_id = pd.DataFrame()

    def prepare_flow(self):
        """Generate f_tr (flow to recycle), f_tw (flow to waste) and f_d (flow of demand) that are use as input of
            the optimisation model
         """
        # f_tr
        if self.region == 'GLO':
            self.f_tr = pd.DataFrame(index=self.outflow.index, columns=self.outflow.columns)
            for ind in self.collection_rate.index:
                self.f_tr.loc[ind] = self.outflow.loc[ind].values * self.collection_rate.loc[ind]
            self.f_tr = self.f_tr.sum(level=[1, 2, 3])
        else:
            self.f_tr = pd.DataFrame(index=self.outflow.loc[self.region, :].index, columns=self.outflow.loc[self.region,].columns)
            for ind in self.collection_rate.loc[self.region].index:
                self.f_tr.loc[ind] = self.outflow.loc[self.region].loc[ind].values * self.collection_rate.loc[self.region].loc[ind]
        # f_tw
        if self.region == 'GLO':
            self.f_tw = pd.DataFrame(index=self.outflow.index, columns=self.outflow.columns)
            for ind in self.collection_rate.index:
                self.f_tw.loc[ind] = self.outflow.loc[ind].values * (1-self.collection_rate.loc[ind])
            self.f_tw = self.f_tw.sum(level=[1, 2, 3])
        else:
            self.f_tw = pd.DataFrame(index=self.outflow.loc[self.region, :].index, columns=self.outflow.loc[self.region,].columns)
            for ind in self.collection_rate.loc[self.region].index:
                self.f_tw.loc[ind] = self.outflow.loc[self.region].loc[ind].values * (1-(self.collection_rate.loc[self.region].loc[ind]))
        if self.region == 'GLO':
            self.f_d = self.inflow.sum(level=[1, 2])
        else:
            self.f_d = self.inflow.loc[self.region]

    def solve_case(self, b):
        """Solve the optimisation for à specific case of and a specific number of bin (b)
        """

        opt = SolverFactory('ipopt')
        # Create model
        #start = time.time()
        model = ConcreteModel()

        # Creation of model index
        model.bins = list(range(1, b + 1))
        model.bins_minus_last = list(range(1, b))
        alloying_el = list(dict.fromkeys(self.f_d.index.get_level_values(1)))
        model.alloying_el = alloying_el
        minor_el = alloying_el.copy()
        minor_el.remove('Al')
        model.minority_el = minor_el
        alloys = list(dict.fromkeys(self.f_d.index.get_level_values(0)))
        model.alloys = alloys
        alloy_w = alloys.copy()
        alloy_w.append('Scrap')
        model.alloys_w = alloy_w
        sectors = list(dict.fromkeys(self.f_tr.index.get_level_values(0)))  ### Changed ###
        model.sectors = sectors  ### Changed ###
        model.time_all = self.f_d.columns

        # Creation of the flow to recycle
        if self.dismantling_condition == 'full':
            f_tr = self.f_tr.sum(level=[1, 2])
            model.f_tr = po.Param(model.alloys, model.alloying_el, model.time_all, within=Any,
                                  initialize={(al, ale, time): f_tr.loc[al, ale][time] for al in model.alloys
                                              for ale in model.alloying_el for time in model.time_all})
        if self.dismantling_condition == 'none':
            f_tr = self.f_tr.sum(level=[0, 2])  ### Changed ###
            model.f_tr = po.Param(model.sectors, model.alloying_el, model.time_all, within=Any,
                                  initialize={(sec, ale, time): f_tr.loc[sec, ale][time] for sec in model.sectors
                                              for ale in model.alloying_el for time in model.time_all})

        # Creation of the demand flow
        model.f_d = po.Param(model.alloys, model.alloying_el, model.time_all, within=Any,
                             initialize={(alloy, ale, time): self.f_d.loc[alloy, ale][time]
                                         for alloy in model.alloys for ale in model.alloying_el for time in
                                         model.time_all})

        imp_primary = self.impact_intensity.loc[model.alloying_el]
        model.imp_primary_rate = po.Param(model.alloying_el, model.time_all, within=Any,
                                          initialize={(ale, time): imp_primary.loc[ale][time] for ale in
                                                      model.alloying_el for time in model.time_all})

        # Creation of impact parameters
        imp_scrap = self.impact_intensity.loc['Scrap']
        model.imp_scrap_rate = po.Param(model.time_all, within=Any,
                                        initialize={(time): imp_scrap.loc[time] for time in model.time_all})

        imp_sort = self.impact_intensity.loc['Sorting']
        model.imp_sort_rate = po.Param(model.time_all, within=Any,
                                       initialize={(time): imp_sort.loc[time] for time in model.time_all})

        imp_rec = self.impact_intensity.loc['Recycling']
        model.imp_rec_rate = po.Param(model.time_all, within=Any,
                                      initialize={(time): imp_rec.loc[time] for time in model.time_all})

        # Creation of alloy composition matrix
        model.theta = po.Param(model.alloys, model.alloying_el,
                               initialize={(alloy, ale): self.theta.loc[alloy, ale]
                                           for alloy in model.alloys for ale in model.alloying_el})

        # Creation of the transfer matrix alpha depending on the sorting condition
        if self.dismantling_condition == 'full':
            model.alpha = Var(model.alloys, model.bins, bounds=(0, 1), initialize=1 / b)
        if self.dismantling_condition == 'none':
            model.alpha = Var(model.sectors, model.bins, bounds=(0, 1), initialize=1 / b)

        # Creation of the transfer matrix beta
        model.beta = Var(model.bins, model.alloys_w, model.time_all, bounds=(0, 1),
                         initialize=1 / len(model.alloys_w))

        # Creation of intermediary variable
        model.f_s = Var(model.bins, model.alloying_el, model.time_all, initialize=0, within=NonNegativeReals, )
        model.f_m = Var(model.alloys_w, model.alloying_el, model.time_all, initialize=0, within=NonNegativeReals, )
        model.f_sw = Var(model.alloys, model.alloying_el, model.time_all, initialize=0, within=NonNegativeReals, )
        model.f_r = Var(model.alloys, model.alloying_el, model.time_all, initialize=0, within=NonNegativeReals, )
        model.f_p = Var(model.alloying_el, model.time_all, initialize=0, within=NonNegativeReals, )
        model.f_tp = Var(model.alloying_el, model.time_all, initialize=0, within=NonNegativeReals, )

        # Creation of constraint
        model.cons_total_primary = Constraint(model.alloying_el, model.time_all, rule=cons_total_primary)
        model.cons_alloy_market_balance = Constraint(model.alloying_el, model.time_all,
                                                     rule=cons_alloy_market_balance)
        model.cons_comp_alloys = Constraint(model.alloys, model.minority_el, model.time_all, rule=cons_comp_alloys)

        model.cons_f_r = Constraint(model.alloys, model.alloying_el, model.time_all, rule=cons_f_r)
        model.cons_f_m = Constraint(model.alloys_w, model.alloying_el, model.time_all, rule=cons_f_m)
        if self.dismantling_condition == 'full':
            model.cons_f_s = Constraint(model.bins, model.alloying_el, model.time_all, rule=cons_f_s_alloys)
            model.cons_alpha = Constraint(model.alloys, rule=cons_alpha)
        if self.dismantling_condition == 'none':
            model.cons_f_s = Constraint(model.bins, model.alloying_el, model.time_all, rule=cons_f_s_sectors)
            model.cons_alpha = Constraint(model.sectors, rule=cons_alpha)

        model.cons_beta = Constraint(model.bins, model.time_all, rule=cons_beta)

        # IMPACTS
        model.imp_f_tp = Var(model.time_all, initialize=0, within=NonNegativeReals, )
        model.imp_scrap = Var(model.time_all, initialize=0, within=NonNegativeReals, )
        model.imp_sort = Var(model.time_all, initialize=0, within=NonNegativeReals, )
        model.imp_rec = Var(model.time_all, initialize=0, within=NonNegativeReals, )

        model.cons_imp_t_p = Constraint(model.time_all, rule=cons_imp_f_t_p)
        model.cons_imp_scrap = Constraint(model.time_all, rule=cons_imp_scrap)
        model.cons_imp_sorting = Constraint(model.time_all, rule=cons_imp_sorting)
        model.cons_imp_recycling = Constraint(model.time_all, rule=cons_imp_recycling)

        model.imp_f_tp_total = Var(initialize=0, within=NonNegativeReals, )
        model.imp_scrap_total = Var(initialize=0, within=NonNegativeReals, )
        model.imp_sort_total = Var(initialize=0, within=NonNegativeReals, )
        model.imp_rec_total = Var(initialize=0, within=NonNegativeReals, )

        model.cons_imp_t_p_total = Constraint(rule=cons_imp_f_t_p_total)
        model.cons_imp_scrap_total = Constraint(rule=cons_imp_scrap_total)
        model.cons_imp_sorting_total = Constraint(rule=cons_imp_sorting_total)
        model.cons_imp_recycling_total = Constraint(rule=cons_imp_recycling_total)

        # Creation of the objective
        model.obj_imp = Objective(rule=obj_imp, sense=minimize)

        # SOLVING
        opt.solve(model)  # , tee=True)

        alpha = todf(model.alpha)
        beta = todf(model.beta)
        f_s = todf(model.f_s)
        f_m = todf(model.f_m)
        f_sw = todf(model.f_sw)
        f_r = todf(model.f_r)
        f_p = todf(model.f_p)
        f_tp = todf(model.f_tp)
        imp_f_tp = todf(model.imp_f_tp)
        imp_scrap = todf(model.imp_scrap)
        imp_sort = todf(model.imp_sort)
        imp_rec = todf(model.imp_rec)
        imp_f_tp_df = pd.DataFrame(imp_f_tp, columns=['Primary production']).T
        imp_scrap_df = pd.DataFrame(imp_scrap, columns=['Scrap']).T
        imp_sort_df = pd.DataFrame(imp_sort, columns=['Sorting']).T
        imp_rec_df = pd.DataFrame(imp_rec, columns=['Recycling']).T
        imp = pd.concat([imp_f_tp_df, imp_sort_df, imp_rec_df, imp_scrap_df])

        #end = time.time()
        #print(end - start)
        return (alpha, beta, f_s, f_m, f_sw, f_r, f_p, f_tp, imp)

    def solve_ideal(self):
        """Solving the optimisation for different number of bins
        """
        f_tp_id = self.f_d - self.f_tr.sum(level=[1, 2])
        f_tp_id[f_tp_id < 0] = 0
        imp_tp_id = f_tp_id.sum(level=1) * self.impact_intensity
        imp_tp_id.dropna(inplace=True)
        imp_tp_id = imp_tp_id.sum()
        imp_tp_id = pd.DataFrame(imp_tp_id, columns=['Primary production']).T

        f_scrap_id = self.f_tr.sum(level=[1, 2]) - self.f_d
        f_scrap_id[f_scrap_id < 0] = 0

        imp_scrap_id = f_scrap_id.sum(axis=0) * self.impact_intensity.loc['Scrap']
        imp_scrap_id = pd.DataFrame(imp_scrap_id, columns=['Scrap']).T

        imp_sorting_id = self.f_tr.sum() * self.impact_intensity.loc['Sorting']
        imp_sorting_id = pd.DataFrame(imp_sorting_id, columns=['Sorting']).T

        imp_recycling_id = self.f_tr.sum() * self.impact_intensity.loc['Recycling']
        imp_recycling_id = pd.DataFrame(imp_recycling_id, columns=['Recycling']).T

        imp_id = pd.concat([imp_tp_id, imp_sorting_id, imp_recycling_id, imp_scrap_id])
        imp_id = pd.concat([imp_id], keys=['Ideal'])
        self.imp_id = imp_id
        self.f_tp_id = f_tp_id
        self.f_scrap_id = f_scrap_id

    def solve(self):
        """Solve for every number of bins
        """
        # Prepare data
        self.prepare_flow()

        #Solve for different number of bins
        for b in list(np.arange(1, self.b_max + 1)):
            print(b)
            # Solve a case
            alpha, beta, f_s, f_m, f_sw, f_r, f_p, f_tp, imp = self.solve_case(b)
            # Concat results for a specific bin into big result matrices
            alpha_b = pd.concat([alpha], keys=[b])
            self.alpha_all = pd.concat([self.alpha_all, alpha_b], axis=0)
            beta_b = pd.concat([beta], keys=[b])
            self.beta_all = pd.concat([self.beta_all, beta_b], axis=0)
            f_s_b = pd.concat([f_s], keys=[b])
            self.f_s_all = pd.concat([self.f_s_all, f_s_b], axis=0)
            f_m_b = pd.concat([f_m], keys=[b])
            self.f_m_all = pd.concat([self.f_m_all, f_m_b], axis=0)
            f_sw_b = pd.concat([f_sw], keys=[b])
            self.f_sw_all = pd.concat([self.f_sw_all, f_sw_b], axis=0)
            f_r_b = pd.concat([f_r], keys=[b])
            self.f_r_all = pd.concat([self.f_r_all, f_r_b], axis=0)
            f_p_b = pd.concat([f_p], keys=[b])
            self.f_p_all = pd.concat([self.f_p_all, f_p_b], axis=0)
            f_tp_b = pd.concat([f_tp], keys=[b])
            self.f_tp_all = pd.concat([self.f_tp_all, f_tp_b], axis=0)
            imp_b = pd.concat([imp], keys=[b])
            self.imp_all = pd.concat([self.imp_all, imp_b], axis=0)

        #Solve ideal case
        self.solve_ideal()

    """ 
    Plot 
    """
    def plot_impacts(self):
        """Plot evolution of the impact of the case over time and for different number of bins
        """
        time = list(self.imp_all.columns)

        colors = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#abd9e9', '#74add1', '#4575b4','#313695']
        fig, (ax1) = plt.subplots(1)
        fig.set_figwidth(8)
        fig.set_figheight(6)

        x = self.imp_all.columns
        for b in self.imp_all.sum(level=0).index:
            y = self.imp_all.sum(level=0).loc[b]  # .values
            position = list(self.imp_all.sum(level=0).index).index(b)
            ax1.plot(x, y, ms=4, lw=2, color=colors[position])

        # Legend
        legend_bins = plt.legend(list(self.imp_all.sum(level=0).index), title='Number of bins', loc=((1.15, 0.2)))
        plt.gca().add_artist(legend_bins)
        ax1.xaxis.set_minor_locator(plt.MultipleLocator(5))

        plt.xlabel('Year')
        if self.impact == 'CC':
            y_label = r'Climate change - [Mt $CO_2$ eq /y]'
        if self.impact == 'RE':
            y_label = 'Mineral resource scarcity - [Mt Cu eq /y]'

        # Plot ideal impact
        ax1.plot(time, self.imp_id.sum(), marker='d', lw=0, c='k')
        ax1.set_ylim(ymin=0)

        # Legends
        ax2 = ax1
        custom_line_ideal = [Line2D([0], [0], marker='d', ls='--', color='k', lw=0)]
        ax2.legend(custom_line_ideal, ['Ideal sorting'], loc=3)

        plt.ylabel(y_label, size=16)
        # Start y axis by 0
        ax1.set_ylim(ymin=0)
        ax1.set_xlim(xmin=2013, xmax=2102)

        ax3 = ax1.twinx()
        # Plot total demand
        ax3.plot(time, self.inflow.sum(), ls=':', ms=4, lw=2, c='k')
        ax3.set_ylim(ymin=0)

        plt.ylabel('Overall demand [Mt/y]', size=16)

        # Legends
        custom_lines = [Line2D([0], [0], ls=':', color='k', lw=2)]
        ax3.legend(custom_lines, ['Demand'], loc=4)

        plt.show()

    def plot_primary_demand(self):
        """Plot the evolution of the primary demand according different number of bins and the total demand
        """
        time = list(self.f_tp_all.columns)
        primary = self.f_tp_all.sum(level=0)
        demand = self.f_d.sum()[time]

        colors = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#abd9e9', '#74add1', '#4575b4','#313695']

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.set_minor_locator(plt.MultipleLocator(5))

        # Plot demand for different numbers of bins
        for b in primary.index:
            y = primary.loc[b]
            position = list(primary.index).index(b)
            plt.plot(time, y, color=colors[position], lw=2)

        # Plot ideal demand

        plt.plot(time, self.f_tp_id.sum(), marker='d', ls='--', color='k', lw=0)
        custom_line_ideal = [Line2D([0], [0], marker='d', ls='--', color='k', lw=0)]
        legend_ideal = plt.legend(custom_line_ideal, ['Ideal sorting'], loc=3)
        legend_id = plt.gca().add_artist(legend_ideal)

        # Plot total demand
        plt.plot(time, demand, ls=':', c='k')

        ax.set_xlim(xmin=2013, xmax=2102)
        ax.set_ylim(ymin=0)
        legend_bin = plt.legend(title='Number of bins', loc=((1.05, 0.2)))
        legend = plt.gca().add_artist(legend_bin)

        # Legends
        custom_lines = [Line2D([0], [0], ls=':', color='k', lw=2)]
        ax.legend(custom_lines, ['Demand'], loc=2)
        plt.xlabel('Year')
        plt.ylabel('Primary consumption [Mt / year]', size=16)
        plt.show()

    def plot_recycled_content(self):
        """Plot the evolution of the recycled content of primary demand according  different number of bins
         """
        time = list(self.f_tp_all.columns)
        primary = self.f_tp_all.sum(level=0)
        demand = self.f_d.sum()

        colors = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#abd9e9', '#74add1', '#4575b4','#313695']

        recycled_content = (demand - primary) / demand * 100

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)

        # Plot demand for different numbers of bins
        for b in recycled_content.index:
            y = recycled_content.loc[b]
            position = list(primary.index).index(b)
            plt.plot(time, y, color=colors[position])

        # PLot ideal case
        recycled_content_id = (demand - self.f_tp_id) / demand * 100
        plt.plot(time, self.f_tp_id.sum(), marker='d', ls='--', color='k', lw=0)
        custom_line_ideal = [Line2D([0], [0], marker='d', ls='--', color='k', lw=0)]
        legend_ideal = plt.legend(custom_line_ideal, ['Ideal sorting'], loc=1)
        legend_id = plt.gca().add_artist(legend_ideal)

        ax.set_xlim(xmin=2013, xmax=2102)
        ax.set_ylim(ymin=0, ymax=100)

        plt.xlabel('Year')
        plt.ylabel('% of recycled content', size=16)

        legend_bins = plt.legend(title='Number of bins', loc=(2))
        legend = plt.gca().add_artist(legend_bins)

    def plot_scrap_gen(self):
        """Plot the evolution of the unused scrap according different number of bins  and flow to recycle
        """

        colors = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#abd9e9', '#74add1', '#4575b4','#313695']

        time = self.f_m_all.columns
        scrap = self.f_m_all.loc(axis=0)[:, 'Scrap', :].sum(level=0)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.set_minor_locator(plt.MultipleLocator(5))

        # Plot unused scrap generation for different numbers of bins
        for b in scrap.index:
            y = scrap.loc[b]
            position = list(scrap.index).index(b)
            plt.plot(time, y, color=colors[position], lw=2)

        # Plot ideal demand
        plt.plot(time, self.f_scrap_id.sum(), marker='d', ls='--', color='k', lw=0)
        custom_line_ideal = [Line2D([0], [0], marker='d', ls='--', color='k', lw=0)]
        legend_ideal = plt.legend(custom_line_ideal, ['Ideal sorting'], loc=2)
        legend_id = plt.gca().add_artist(legend_ideal)

        # Plot flow to recycle
        plt.plot(time, self.f_tr.sum()[time], ls=':', c='k')

        ax.set_xlim(xmin=2013, xmax=2102)
        ax.set_ylim(ymin=0)
        legend_bins = plt.legend(title='Number of bins', loc=((1.05, 0.2)))
        legend = plt.gca().add_artist(legend_bins)

        # Legends
        custom_lines = [Line2D([0], [0], ls=':', color='k', lw=2)]
        ax.legend(custom_lines, ['Flow to recycle'], loc=1)
        plt.xlabel('year')
        plt.ylabel('Unused scrap generated [Mt]', size=16)
        plt.show()

    def plot_element_composition_f_sw_f_p(self, n, y):
        """Plot and compare the element composition of priamry demand and sweetening flow for a specific year and a specific number of bins
          Args:
             -----
                n  : [int] specific number of bin to plot
                y  : [int] specific year
         """
        sweetening_proportion = self.f_sw_all.loc[2][2050].sum(level=1) / self.f_sw_all.loc[2][2050].sum().sum() * 100
        primary_proportion = (self.f_p_all.loc[n][y] / self.f_p_all.loc[n][y].sum()) * 100
        proportion = pd.concat([primary_proportion, sweetening_proportion], axis=1)
        proportion.columns = ['Primary', 'Sweetening', ]

        y_Al = proportion.loc['Al']
        y_Cr = proportion.loc['Cr']
        y_Cu = proportion.loc['Cu']
        y_Fe = proportion.loc['Fe']
        y_Mg = proportion.loc['Mg']
        y_Mn = proportion.loc['Mn']
        y_Ni = proportion.loc['Ni']
        y_Si = proportion.loc['Si']
        y_Ti = proportion.loc['Ti']
        y_Zn = proportion.loc['Zn']

        x = proportion.columns

        # plot bars in stack manner
        plt.bar(x, y_Al)
        plt.bar(x, y_Cr, bottom=y_Al)
        plt.bar(x, y_Cu, bottom=y_Al + y_Cr)
        plt.bar(x, y_Fe, bottom=y_Al + y_Cr + y_Cu)
        plt.bar(x, y_Mg, bottom=y_Al + y_Cr + y_Cu + y_Fe)
        plt.bar(x, y_Mn, bottom=y_Al + y_Cr + y_Cu + y_Fe + y_Mg)
        plt.bar(x, y_Ni, bottom=y_Al + y_Cr + y_Cu + y_Fe + y_Mg + y_Mn)
        plt.bar(x, y_Si, bottom=y_Al + y_Cr + y_Cu + y_Fe + y_Mg + y_Mn + y_Ni)
        plt.bar(x, y_Ti, bottom=y_Al + y_Cr + y_Cu + y_Fe + y_Mg + y_Mn + y_Ni + y_Si)
        plt.bar(x, y_Zn, bottom=y_Al + y_Cr + y_Cu + y_Fe + y_Mg + y_Mn + y_Ni + y_Si + y_Ti)

        plt.ylabel("%")
        plt.ylim(ymin=round(min(y_Al)) - 3, ymax=100.5)
        plt.legend(list(proportion.index), loc=(1.05, 0.2))
        plt.title(str(n) + ' bins - ' + str(y))
        plt.show()

    def plot_heat_map_alpha(self, n):
        """Plot the alpha of a specific year and number of bin as a heat map
             -----
                n  : [int] specific number of bin to plot
                y  : [int] specific year
         """

        df = self.alpha_all.loc[n] * 100
        plt.pcolor(df[np.arange(n) + 1])
        plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
        plt.xticks(np.arange(0.5, n, 1), df.columns[np.arange(n)])
        plt.colorbar()

    def plot_sankey(self, n_bins, year):
        """Plot Sankey of optimised system
             -----
                n_bins  : [int] specific number of bin to plot
                year  : [int] specific year
         """
        if self.dismantling_condition == 'full':

            ##########
            # Labels #
            ##########
            label_sector = list(dict.fromkeys(list(self.f_tr.index.get_level_values(level=0))))  # 6 [0,1,2,3,4,5]

            label_waste = ['Waste']  # 1 - [6]
            label_alloy = list(dict.fromkeys(list(self.f_tr.index.get_level_values(level=1))))  # 16 - [7,8,9,...22]
            label_bin = list(np.arange(1, n_bins + 1))  # n_bins - [23, ... , 23+n_bins-1]
            label_alloy_r = []  # 16 - [23+n_bins, ... 38+n_bins]
            for i in label_alloy:
                alloy_r_temp = i + '_r'
                label_alloy_r.append(alloy_r_temp)
            label_scrap = ['Unused scrap']  # 1 - [39+n_bins]
            label_sweetening = ['Sweetening']  # 1 - [40+n_bins]

            label = label_sector + label_waste + label_alloy + label_bin + label_alloy_r + label_scrap + label_sweetening

            ##########
            # Values #
            ##########

            # Calculation
            values_alpha = self.alpha_all.loc[n_bins, :].mul(self.f_tr[year].sum(level=1), axis=0) * 1000
            values_alpha[values_alpha < 1] = 0

            values_beta = self.beta_all.loc[n_bins, :][year].mul(self.f_s_all.loc[n_bins][year].sum(axis=0, level=0),
                                                                 axis=0, level=0) * 1000
            values_beta[values_beta < 1] = 0

            values_sw = self.f_sw_all.loc[n_bins][2015].sum(level=0) * 1000
            values_sw[values_sw < 1] = 0

            # 1-Sectors to alloys
            list_value_s_a = []
            for sec in label_sector:
                s_a_sec = self.f_tr.loc(axis=1)[year].sum(level=[0, 1]) * 1000
                list_value_s_a_sec = [dict(s_a_sec[sec]).get(alloy) for alloy in label_alloy]
                list_value_s_a.append(list_value_s_a_sec)

            list_value_s_a = [item for sublist in list_value_s_a for item in sublist]

            # 2-Sector to waste
            s_w = self.f_tw.sum(level=[0])[year] * 1000
            list_value_s_w = [s_w.get(sector) for sector in label_sector]

            # 3-Alloys to bins
            dict_a_b = {}
            for i in list(np.arange(1, n_bins + 1)):
                dict_a_b['a_b' + str(i)] = dict(values_alpha.loc(axis=1)[i])
            list_value_a_b = []
            for i in list(np.arange(1, n_bins + 1)):
                dict_a_b_bin = dict_a_b['a_b' + str(i)]
                list_value_a_b_bin = [dict_a_b_bin.get(alloy) for alloy in label_alloy]
                list_value_a_b.append(list_value_a_b_bin)
            list_value_a_b = [item for sublist in list_value_a_b for item in sublist]

            # 4-bins to recycled alloys
            dict_b_ar = {}
            for i in list(np.arange(1, n_bins + 1)):
                dict_b_ar['b' + str(i) + '_ar'] = dict(values_beta.loc(axis=0)[i])

            list_value_b_ar = []
            for i in list(np.arange(1, n_bins + 1)):
                dict_b_ar_bin = dict_b_ar['b' + str(i) + '_ar']
                list_value_b_ar_bin = [dict_b_ar_bin.get(alloy) for alloy in label_alloy]

                list_value_b_ar.append(list_value_b_ar_bin)
            list_value_b_ar = [item for sublist in list_value_b_ar for item in sublist]

            # 5-bins to unused scrap
            list_value_b_sc = []
            for i in list(np.arange(1, n_bins + 1)):
                dict_b_sc_bin = dict_b_ar['b' + str(i) + '_ar']
                list_value_b_sc_bin = [dict_b_sc_bin.get('Scrap')]

                list_value_b_sc.append(list_value_b_sc_bin)
            list_value_b_sc = [item for sublist in list_value_b_sc for item in sublist]

            # 6-sweetening to alloys_r

            sw_ar = self.f_sw_all.loc[n_bins][2015].sum(level=0) * 1000
            sw_ar[sw_ar < 1] = 0

            list_value_sw_ar = [dict(sw_ar).get(alloy) for alloy in label_alloy]

            ##########
            # Source #
            ##########

            # 1-Sectors to alloys
            list_source_s_a = []
            for s in np.arange(0, len(label_sector)):
                list_source_s_a_temp = list([s]) * len(label_alloy)
                list_source_s_a.append(list_source_s_a_temp)
            list_source_s_a = [item for sublist in list_source_s_a for item in sublist]

            # 2-Sector to waste
            list_source_s_w = list(np.arange(0, len(label_sector))) * len(label_waste)

            # 3-Alloys to bins
            start_alloy = len(label_sector) + len(label_waste)
            list_source_a_b = list(np.arange(start_alloy, start_alloy + len(label_alloy))) * len(label_bin)

            # 4-bins to recycled alloys
            start_bin = len(label_sector) + len(label_waste) + len(label_alloy)
            list_source_b_ar = []
            for b in label_bin:
                list_source_b_ar_temp = list([start_bin - 1 + b] * len(label_alloy))
                list_source_b_ar.append(list_source_b_ar_temp)
            list_source_b_ar = [item for sublist in list_source_b_ar for item in sublist]

            # 5-bins to unused scrap
            start_bin = len(label_sector) + len(label_waste) + len(label_alloy)
            list_source_b_sc = []
            for b in label_bin:
                list_source_b_sc_temp = list([start_bin - 1 + b] * len(label_scrap))
                list_source_b_sc.append(list_source_b_sc_temp)
            list_source_b_sc = [item for sublist in list_source_b_sc for item in sublist]

            # 6-sweetening to alloys_r
            list_source_sw_ar = list([len(label) - 1]) * len(label_alloy_r)

            ##########
            # Target #
            ##########

            # Target
            # 1-Sectors to alloys
            start_alloy = len(label_sector) + len(label_waste)
            list_target_s_a = list(np.arange(0, len(label_alloy)) + start_alloy) * len(label_sector)

            # 2-Sector to waste
            start_waste = len(label_sector)
            list_target_s_w = [start_waste] * len(label_sector)

            # 3-Alloys to bins
            start_bin = len(label_sector) + len(label_waste) + len(label_alloy)
            list_target_a_b = []
            for b in np.arange(0, len(label_bin)):
                list_target_a_b_temp = [start_bin + b] * len(label_alloy)
                list_target_a_b.append(list_target_a_b_temp)
            list_target_a_b = [item for sublist in list_target_a_b for item in sublist]

            # 4-bins to recycled alloys

            start_alloy_r = len(label_sector) + len(label_waste) + len(label_alloy) + len(label_bin)
            list_target_b_ar = list(np.arange(start_alloy_r, start_alloy_r + len(label_alloy_r))) * len(label_bin)

            # 5-bins to unused scrap
            start_scrap = len(label_sector) + len(label_waste) + len(label_alloy) + len(label_bin) + len(
                label_alloy_r)
            list_target_b_sc = [start_scrap] * len(label_bin)

            # 6-sweetening to alloys_r
            start_alloy_r = len(label_sector) + len(label_waste) + len(label_alloy) + len(label_bin)
            list_target_sw_ar = list(np.arange(start_alloy_r, start_alloy_r + len(label_alloy)))

            ########
            # Plot #
            ########

            list_value = list_value_s_a + list_value_s_w + list_value_a_b + list_value_b_ar + list_value_b_sc + list_value_sw_ar
            list_source = list_source_s_a + list_source_s_w + list_source_a_b + list_source_b_ar + list_source_b_sc + list_source_sw_ar
            list_target = list_target_s_a + list_target_s_w + list_target_a_b + list_target_b_ar + list_target_b_sc + list_target_sw_ar

            color_sector = ['#2166ac']
            color_waste = ['#b2182b']
            color_bin = ['#636363']
            color_alloy = ['#4393c3']
            color_scrap = ['#b2182b']
            color_sweetening = ['#92c5de']
            color_link = '#bdbdbd'

            color_sector = color_sector * len(label_sector)
            color_bin = color_bin * len(label_bin)
            color_alloy = color_alloy * len(label_alloy)
            color_nod = color_sector + color_waste + color_alloy + color_bin + color_alloy + color_scrap + color_sweetening

            fig = go.Figure(data=[go.Sankey(
                node=dict(pad=15, thickness=40, line=dict(color="black", width=0.5), label=label, color=color_nod),
                link=dict(source=list_source, target=list_target, value=list_value, color=color_link))])

            fig.update_layout(title_text=str(year) + ' - ' + str(n_bins) + ' bins', font_size=12)
            fig.show()

        if self.dismantling_condition == 'none':

            ##########
            # Labels #
            ##########
            label_sector = list(
                dict.fromkeys(list(self.f_tr.index.get_level_values(level=0))))  # 6 [0,1,2,3,4,5]

            label_waste = ['Waste']  # 1 - [6]
            label_alloy = list(dict.fromkeys(list(self.f_tr.index.get_level_values(level=1))))
            label_bin = list(np.arange(1, n_bins + 1))  # n_bins - [7, ... , 7+n_bins-1]
            label_alloy_r = []  # 16 - [7+n_bins, ... 22+n_bins]
            for i in label_alloy:
                alloy_r_temp = i + '_r'
                label_alloy_r.append(alloy_r_temp)
            label_scrap = ['Unused scrap']  # 1 - [23+n_bins]
            label_sweetening = ['Sweetening']  # 1 - [24+n_bins]

            label = label_sector + label_waste + label_bin + label_alloy_r + label_scrap + label_sweetening
            # print(label)

            ##########
            # Values #
            ##########

            # Calculation
            values_alpha = self.alpha_all.loc[n_bins, :].mul(self.f_tr[year].sum(level=0), axis=0) * 1000
            values_alpha[values_alpha < 1] = 0

            values_beta = self.beta_all.loc[n_bins, :][year].mul(self.f_s_all.loc[n_bins][year].sum(axis=0, level=0),
                                                                 axis=0, level=0) * 1000
            values_beta[values_beta < 1] = 0

            values_sw = self.f_sw_all.loc[n_bins][2015].sum(level=1) * 1000
            values_sw[values_sw < 1] = 0

            # 1-Sectors to bins
            dict_s_b = {}
            for i in list(np.arange(1, n_bins + 1)):
                dict_s_b['s_b' + str(i)] = dict(values_alpha.loc(axis=1)[i])

            list_value_s_b = []
            for i in list(np.arange(1, n_bins + 1)):
                dict_s_b_bin = dict_s_b['s_b' + str(i)]
                list_value_s_b_bin = [dict_s_b_bin.get(sec) for sec in label_sector]
                list_value_s_b.append(list_value_s_b_bin)
            list_value_s_b = [item for sublist in list_value_s_b for item in sublist]

            # 2-Sector to waste
            s_w = self.f_tw.sum(level=[0])[year] * 1000
            list_value_s_w = [s_w.get(sector) for sector in label_sector]

            # 3-bins to recycled alloys
            dict_b_ar = {}
            for i in list(np.arange(1, n_bins + 1)):
                dict_b_ar['b' + str(i) + '_ar'] = dict(values_beta.loc(axis=0)[i])

            list_value_b_ar = []
            for i in list(np.arange(1, n_bins + 1)):
                dict_b_ar_bin = dict_b_ar['b' + str(i) + '_ar']
                list_value_b_ar_bin = [dict_b_ar_bin.get(alloy) for alloy in label_alloy]
                list_value_b_ar.append(list_value_b_ar_bin)
            list_value_b_ar = [item for sublist in list_value_b_ar for item in sublist]

            # 4-bins to unused scrap
            list_value_b_sc = []
            for i in list(np.arange(1, n_bins + 1)):
                dict_b_sc_bin = dict_b_ar['b' + str(i) + '_ar']
                list_value_b_sc_bin = [dict_b_sc_bin.get('Scrap')]
                list_value_b_sc.append(list_value_b_sc_bin)
            list_value_b_sc = [item for sublist in list_value_b_sc for item in sublist]

            # 5-sweetening to alloys_r

            sw_ar = self.f_sw_all.loc[n_bins][2015].sum(level=0) * 1000
            sw_ar[sw_ar < 1] = 0
            list_value_sw_ar = [dict(sw_ar).get(alloy) for alloy in label_alloy]

            ##########
            # Source #
            ##########

            # 1-Sectors to bins
            list_source_s_b = list(np.arange(0, len(label_sector))) * len(label_bin)

            # 2-Sector to waste
            list_source_s_w = list(np.arange(0, len(label_sector))) * len(label_waste)

            # 3-bins to recycled alloys
            start_bin = len(label_sector) + len(label_waste)
            list_source_b_ar = []
            for b in label_bin:
                list_source_b_ar_temp = list([start_bin - 1 + b] * len(label_alloy))
                list_source_b_ar.append(list_source_b_ar_temp)
            list_source_b_ar = [item for sublist in list_source_b_ar for item in sublist]

            # 4-bins to unused scrap
            start_bin = len(label_sector) + len(label_waste)
            list_source_b_sc = []
            for b in label_bin:
                list_source_b_sc_temp = list([start_bin - 1 + b] * len(label_scrap))
                list_source_b_sc.append(list_source_b_sc_temp)
            list_source_b_sc = [item for sublist in list_source_b_sc for item in sublist]

            # 5-sweetening to alloys_r
            list_source_sw_ar = list([len(label) - 1]) * len(label_alloy_r)

            ##########
            # Target #
            ##########

            # 1-Sectors to bins
            start_bin = len(label_sector) + len(label_waste)
            list_target_s_b = []
            for b in np.arange(0, len(label_bin)):
                list_target_s_b_temp = [start_bin + b] * len(label_sector)
                list_target_s_b.append(list_target_s_b_temp)
            list_target_s_b = [item for sublist in list_target_s_b for item in sublist]

            # 2-Sector to waste
            start_waste = len(label_sector)
            list_target_s_w = [start_waste] * len(label_sector)

            # 3-bins to recycled alloys
            start_alloy_r = len(label_sector) + len(label_waste) + len(label_bin)
            list_target_b_ar = list(np.arange(start_alloy_r, start_alloy_r + len(label_alloy_r))) * len(label_bin)

            # 4-bins to unused scrap
            start_scrap = len(label_sector) + len(label_waste) + len(label_bin) + len(label_alloy_r)
            list_target_b_sc = [start_scrap] * len(label_bin)

            # 5-sweetening to alloys_r
            start_alloy_r = len(label_sector) + len(label_waste) + len(label_bin)
            list_target_sw_ar = list(np.arange(start_alloy_r, start_alloy_r + len(label_alloy)))

            ########
            # Plot #
            ########

            list_value = list_value_s_b + list_value_s_w + list_value_b_ar + list_value_b_sc + list_value_sw_ar
            list_source = list_source_s_b + list_source_s_w + list_source_b_ar + list_source_b_sc + list_source_sw_ar
            list_target = list_target_s_b + list_target_s_w + list_target_b_ar + list_target_b_sc + list_target_sw_ar

            color_sector = ['#2166ac'] * len(label_sector)
            color_waste = ['#b2182b']
            color_bin = ['#636363'] * len(label_bin)
            color_alloy = ['#4393c3'] * len(label_alloy)
            color_scrap = ['#b2182b']
            color_sweetening = ['#92c5de']
            color_link = '#bdbdbd'
            color_nod = color_sector + color_waste + color_bin + color_alloy + color_scrap + color_sweetening

            fig = go.Figure(data=[go.Sankey(
                node=dict(pad=15, thickness=40, line=dict(color="black", width=0.5), color=color_nod, label=label),
                link=dict(source=list_source, target=list_target, value=list_value, color=color_link))])

            fig.update_layout(title_text=str(year) + ' - ' + str(n_bins) + ' bins', font_size=12)
            fig.show()

""" 
CONSTRAINTS
"""
# FLOWS
def cons_f_s_sectors(model, bins, alloying_el, time):
    # Constraint on f_s composition (flow after first transfer matrix - alpha)
    return model.f_s[(bins, alloying_el, time)] == sum(
        model.f_tr[(sector, alloying_el, time)] * model.alpha[(sector, bins)] for sector in model.sectors)

def cons_f_s_alloys(model, bins, alloying_el, time):
    # Constraint on f_s composition (flow after first transfer matrix - alpha)
    return model.f_s[(bins, alloying_el, time)] == sum(
        model.f_tr[(alloy, alloying_el, time)] * model.alpha[(alloy, bins)] for alloy in model.alloys)

def cons_alpha(model, line):
    # Constraint on alpha matrix. Sum of the line = 1
    return sum(model.alpha[(line, n)] for n in model.bins) == 1

def prod(model, line):
    # Constraint on alpha matrix. Sum of the line = 1
    return sum(model.alpha[(line, n)] for n in model.bins) == 1

def cons_f_m(model, alloys, alloying_el, time):
    # Constraint on f_m composition (flow after second transfer matrix - beta)
    return model.f_m[(alloys, alloying_el, time)] == sum(
        model.f_s[(n, alloying_el, time)] * model.beta[(n, alloys, time)] for n in model.bins)

def cons_beta(model, n, time):
    # Constraint on beta matrix. Sum of the line = 1
    return sum(model.beta[(n, j, time)] for j in model.alloys_w) == 1

def cons_f_r(model, alloys, alloying_el, time):
    # Constraint on f_r composition (flow after sweetening)
    return model.f_r[(alloys, alloying_el, time)] == model.f_m[(alloys, alloying_el, time)] + model.f_sw[
        (alloys, alloying_el, time)]

def cons_comp_alloys(model, alloy, minority_el, time):
    # Constraint on composition of different alloys of f_r
    def tot_mass_alloy(j, time):
        return sum(model.f_r[j, i, time] for i in model.alloying_el)

    return model.f_r[alloy, minority_el, time] == model.theta[(alloy, minority_el)] * tot_mass_alloy(alloy,
                                                                                                     time)

def cons_alloy_market_balance(model, alloy_el, time):
    # Constraint on substitution between primary and recycling flow to supply the demand
    return model.f_p[alloy_el, time] == sum(
        model.f_d[(j, alloy_el, time)] - model.f_r[(j, alloy_el, time)] for j in model.alloys)

def cons_total_primary(model, alloying_el, time):
    # Constraint calculating the total primary consumption
    return model.f_tp[alloying_el, time] == model.f_p[alloying_el, time] + sum(
        model.f_sw[j, alloying_el, time] for j in model.alloys)

# IMPACTS

def cons_imp_f_t_p(model, time):
    # Constraint calculating the impacts of primary production
    return model.imp_f_tp[time] == sum(
        model.f_tp[(ale, time)] * model.imp_primary_rate[(ale, time)] for ale in model.alloying_el)

def cons_imp_scrap(model, time):
    # Constraint calculating the impacts of scrap generation
    return model.imp_scrap[time] == sum(model.f_m[('Scrap', ale, time)] for ale in model.alloying_el) * \
           model.imp_scrap_rate[time]

def cons_imp_sorting(model, time):
    # Constraint calculating the impacts of sorting
    return model.imp_sort[time] == sum(
        model.f_s[(b, ale, time)] for b in model.bins for ale in model.alloying_el) * model.imp_sort_rate[time]

def cons_imp_recycling(model, time):
    # Constraint calculating the impacts of recycling
    return model.imp_rec[time] == sum(
        model.f_s[(b, ale, time)] for b in model.bins for ale in model.alloying_el) * model.imp_rec_rate[time]

def cons_imp_f_t_p_total(model):
    # Constraint calculating the total impacts of primary production
    return model.imp_f_tp_total == sum(model.imp_f_tp[time] for time in model.time_all)

def cons_imp_scrap_total(model):
    # Constraint calculating the total impacts of scrap
    return model.imp_scrap_total == sum(model.imp_scrap[time] for time in model.time_all)

def cons_imp_sorting_total(model):
    # Constraint calculating the total impacts of primary production
    return model.imp_sort_total == sum(model.imp_sort[time] for time in model.time_all)

def cons_imp_recycling_total(model):
    # Constraint calculating the total impacts of primary production
    return model.imp_rec_total == sum(model.imp_rec[time] for time in model.time_all)

    """ 
    OBJECTIVE
    """

def obj_imp(model):
        # Objective to minimize impacts
        return (model.imp_f_tp_total + model.imp_scrap_total + model.imp_sort_total + model.imp_rec_total)

""" 
EXTRA functions
"""
def todf(data):
    """ Simple function to transform pyomo element as Pandas DataFrame"""
    try:
        out = pd.Series(data.get_values())
    except AttributeError:
        # probably already is a dataframe
        out = data

    if out.index.nlevels > 1:
        out = out.unstack()
    return out
