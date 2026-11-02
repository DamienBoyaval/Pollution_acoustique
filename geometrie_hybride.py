"""Professor fractal wall plus a side-branch cavity/neck from the earlier project."""
import numpy as np
from scipy.ndimage import binary_dilation, label
import _env
import comparaison_materiaux as comp
import preprocessing as pre

CROSS=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=bool)


def build_problem(N=128,level=1,resonator=True,quantity=.4,length_m=1.,sound_speed=340.,
                  neck_length=.0625,neck_width=.03125,cavity_width=.0625,
                  cavity_height=.0625,center_y=.375):
    """Dimensions are normalized to L. New cavity walls are rigid Neumann walls.

    The original top Dirichlet source and bottom Robin material region remain.
    The mouth opens a small part of the rigid right wall into the added air.
    """
    if not np.isfinite(length_m) or length_m<=0 or quantity<=0 or not np.isfinite(quantity):
        raise ValueError('Positive finite length and quantity required.')
    if not np.isfinite(sound_speed) or sound_speed<=0:
        raise ValueError('Positive finite sound speed required.')
    dimensions=[neck_length,neck_width,cavity_width,cavity_height,center_y]
    if not np.all(np.isfinite(dimensions)) or min(dimensions)<=0:
        raise ValueError('Positive finite resonator dimensions required.')
    cells=np.asarray(dimensions)*N
    if not np.allclose(cells,np.round(cells),atol=1e-9,rtol=0):
        raise ValueError('Resonator dimensions must align with the grid; use N=128 or a multiple.')
    nl,nw,cw,ch,cy=np.round(cells).astype(int)
    if nw<4 or cw<4 or ch<4 or nl<4 or nw%2 or ch%2 or nw>=ch:
        raise ValueError('Resolve the neck by >=4 grid intervals; neck/cavity heights must be even and neck narrower.')
    original=comp.make_problem(N,level)
    old,h,old_args,raw_chi=original
    mouth=N-1
    shoulder=mouth+nl
    right=shoulder+cw
    width=right+2
    domain=np.full((2*N,width),_env.NODE_COMPLEMENTARY,dtype=old.dtype)
    domain[:,:N]=old
    main_mask=np.zeros_like(domain,dtype=bool)
    main_mask[:,:N]=old==_env.NODE_INTERIOR
    y,x=np.indices(domain.shape)
    neck=(x>=mouth)&(x<=shoulder)&(y>cy-nw//2)&(y<cy+nw//2)
    cavity=(x>shoulder)&(x<right)&(y>cy-ch//2)&(y<cy+ch//2)
    addition=neck|cavity
    if cy-ch//2<1 or cy+ch//2>=domain.shape[0]-1:
        raise ValueError('Cavity must remain away from outer grid limits.')
    if np.any(old[cy-nw//2+1:cy+nw//2,mouth]!=_env.NODE_NEUMANN):
        raise ValueError('The neck must open into the rigid lateral wall, not into Robin or Dirichlet nodes.')
    if resonator:
        if np.any(np.isin(domain[addition],[_env.NODE_ROBIN,_env.NODE_DIRICHLET])):
            raise ValueError('Resonator would overwrite an absorbing or source boundary.')
        domain[addition]=_env.NODE_INTERIOR
        outline=binary_dilation(addition,structure=CROSS)&~addition
        domain[outline&(domain==_env.NODE_COMPLEMENTARY)]=_env.NODE_NEUMANN
    interior=domain==_env.NODE_INTERIOR
    _,components=label(interior,structure=CROSS)
    if components!=1:
        raise ValueError('The air domain must be connected to the source.')
    if np.any(interior[0]) or np.any(interior[-1]) or np.any(interior[:,0]) or np.any(interior[:,-1]):
        raise ValueError('An interior node touches the edge of the computational grid.')
    if np.any(binary_dilation(interior,structure=CROSS)&(domain==_env.NODE_COMPLEMENTARY)):
        raise ValueError('An air node has an unclassified exterior neighbour.')
    # Reuse the same professor coefficients, with padding for the lateral branch.
    bp,ap,ad,bn,ar,br=pre._set_coefficients_of_pde(*domain.shape)
    f,fd,fn,fr=pre._set_rhs_of_pde(*domain.shape)
    f[:,:N],fd[:,:N],fn[:,:N],fr[:,:N]=old_args[:4]
    seed=np.zeros(domain.shape,dtype=float)
    seed[:,:N]=raw_chi
    robin=domain==_env.NODE_ROBIN
    volume=quantity/(h*length_m*np.count_nonzero(robin))
    if volume>1:
        raise ValueError('Quantity exceeds available Robin material region.')
    chi=comp.demo.your_project_density(seed,domain,volume)
    np.testing.assert_array_equal(domain[:,:N]==_env.NODE_ROBIN,old==_env.NODE_ROBIN)
    np.testing.assert_array_equal(domain[:,:N]==_env.NODE_DIRICHLET,old==_env.NODE_DIRICHLET)
    area=(cavity_width*length_m)*(cavity_height*length_m)
    estimated=sound_speed/(2*np.pi)*np.sqrt((neck_width*length_m)/(area*neck_length*length_m))
    metadata={'N':N,'level':level,'resonator':bool(resonator),'length_m':length_m,
              'mouth_x_normalized':mouth*h,'center_y_normalized':center_y,
              'neck_length_normalized':neck_length,'neck_width_normalized':neck_width,
              'cavity_width_normalized':cavity_width,'cavity_height_normalized':cavity_height,
              'neck_intervals_across_width':int(nw),'quantity_discrete':float(h*length_m*chi.sum()),
              'helmholtz_estimate_hz_no_end_correction':float(estimated),
              'main_area_normalized':float(main_mask.sum()*h*h),
              'total_area_normalized':float(interior.sum()*h*h),
              'robin_nodes':int(robin.sum()),'air_components':int(components)}
    return (domain,h,(f,fd,fn,fr,bp,ap,ad,bn,br),chi),main_mask,metadata
