/*----------------------------------------------------------------------*/
/*!
 \file porofluid_variablemanager.cpp

 \brief variable manager class for poro multiphase fluid element

   \level 3

   \maintainer  Anh-Tu Vuong
                vuong@lnm.mw.tum.de
                http://www.lnm.mw.tum.de
                089 - 289-15251
 *----------------------------------------------------------------------*/


#include "porofluid_variablemanager.H"

#include "porofluidmultiphase_ele_parameter.H"

#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_utils.H"


/*----------------------------------------------------------------------*
 | factory method                                           vuong 08/16 |
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
Teuchos::RCP< DRT::ELEMENTS::POROFLUIDMANAGER::VariableManagerInterface<nsd,nen> >
DRT::ELEMENTS::POROFLUIDMANAGER::VariableManagerInterface<nsd,nen>::CreateVariableManager(
    const DRT::ELEMENTS::PoroFluidMultiPhaseEleParameter& para,
    const POROFLUIDMULTIPHASE::Action&     action,
    int numdofpernode)
{
  Teuchos::RCP< VariableManagerInterface<nsd,nen> > varmanager = Teuchos::null;

  // determine action
  switch(action)
  {
  // calculate true pressures and saturation
  case POROFLUIDMULTIPHASE::calc_pres_and_sat:
  // calculate solid pressure
  case POROFLUIDMULTIPHASE::calc_solidpressure:
  {
    // only phi values are needed
    varmanager = Teuchos::rcp(new VariableManagerPhi<nsd,nen>(numdofpernode));
    break;
  }
  // reconstruct velocities
  case POROFLUIDMULTIPHASE::recon_flux_at_nodes:
  {
    // state vector and gradients are needed
    varmanager = Teuchos::rcp(new VariableManagerPhiGradPhi<nsd,nen>(numdofpernode));

    // add manager for displacements and solid velocities in case of ALE
    if(para.IsAle())
      varmanager = Teuchos::rcp(new VariableManagerStruct<nsd,nen>(para.NdsVel(),para.NdsDisp(),varmanager));
    break;
  }
  default:
  {
    // default: potentially read everything
    varmanager = Teuchos::rcp(new VariableManagerPhiGradPhi<nsd,nen>(numdofpernode));

    if(not para.IsStationary())
     varmanager = Teuchos::rcp(new VariableManagerInstat<nsd,nen>(varmanager));

    if(para.IsAle())
      varmanager = Teuchos::rcp(new VariableManagerStruct<nsd,nen>(para.NdsVel(),para.NdsDisp(),varmanager));

    break;
  }
  } // switch(action)

  // if there are other scalar values (from ScaTra coupling) add another manager
  if(para.HasScalar())
    varmanager = Teuchos::rcp(new VariableManagerScalar<nsd,nen>(para.NdsScalar(),varmanager));

  // done
  return varmanager;
};

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 | extract node values related to the state vector 'phinp'   vuong 09/16 |
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void DRT::ELEMENTS::POROFLUIDMANAGER::VariableManagerPhi<nsd,nen>::ExtractElementAndNodeValues(
    const DRT::Element& ele,
    const DRT::Discretization& discretization,
    DRT::Element::LocationArray& la,
    LINALG::Matrix<nsd, nen>& xyze)
{
  // extract local values from the global vectors
  Teuchos::RCP<const Epetra_Vector> phinp = discretization.GetState("phinp");
  if (phinp == Teuchos::null)
    dserror("Cannot get state vector 'phinp'");

   //values of fluid field are always in first dofset
  const std::vector<int>& lm = la[0].lm_;

  // extract element vector from global vector
  DRT::UTILS::ExtractMyValues<LINALG::Matrix<nen,1> >(*phinp,ephinp_,lm);

  // set flag
  this->isextracted_ = true;
  return;
};

/*----------------------------------------------------------------------*
 | evaluate state vector at gauss point                      vuong 09/16 |
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void DRT::ELEMENTS::POROFLUIDMANAGER::VariableManagerPhi<nsd,nen>::EvaluateGPVariables(
    const LINALG::Matrix<nen,1>&   funct,     //! array for shape functions
    const LINALG::Matrix<nsd,nen>& derxy      //! array for shape function derivatives w.r.t x,y,z
    )
{
  // check
  if (not this->isextracted_)
    dserror("ExtractElementAndNodeValues() has not been called!");

  // loop over DOFs
  for (int k = 0; k < this->numdofpernode_; ++k)
  {
    // calculate scalar at t_(n+1) or t_(n+alpha_F)
    phinp_[k] = funct.Dot(ephinp_[k]);
  }

  //done
  this->isevaluated_=true;

  return;
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 | evaluate state vector at gauss point                      vuong 09/16 |
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void DRT::ELEMENTS::POROFLUIDMANAGER::VariableManagerPhiGradPhi<nsd,nen>::EvaluateGPVariables(
    const LINALG::Matrix<nen,1>&   funct,     //! array for shape functions
    const LINALG::Matrix<nsd,nen>& derxy      //! array for shape function derivatives w.r.t x,y,z
    )
{
  // loop over DOFs
  for (int k = 0; k < this->numdofpernode_; ++k)
  {
    // spatial gradient of current scalar value
    gradphi_[k].Multiply(derxy,this->ephinp_[k]);
  }

  // call base class
  VariableManagerPhi<nsd,nen>::EvaluateGPVariables(funct,derxy);

  return;
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 | extract node values related to time derivatives           vuong 09/16 |
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void DRT::ELEMENTS::POROFLUIDMANAGER::VariableManagerInstat<nsd,nen>::ExtractElementAndNodeValues(
    const DRT::Element& ele,
    const DRT::Discretization& discretization,
    DRT::Element::LocationArray& la,
    LINALG::Matrix<nsd, nen>& xyze)
{
  // extract local values from the global vectors
  Teuchos::RCP<const Epetra_Vector> hist = discretization.GetState("hist");
  Teuchos::RCP<const Epetra_Vector> phidtnp = discretization.GetState("phidtnp");
  if (phidtnp==Teuchos::null) dserror("Cannot get state vector 'phidtnp'");

  //values of fluid field are always in first dofset
  const std::vector<int>&    lm = la[0].lm_;

  // extract values from global vector
  DRT::UTILS::ExtractMyValues<LINALG::Matrix<nen,1> >(*hist,ehist_,lm);
  DRT::UTILS::ExtractMyValues<LINALG::Matrix<nen,1> >(*phidtnp,ephidtnp_,lm);

  // call wrapped class
  this->varmanager_->ExtractElementAndNodeValues(ele,discretization,la,xyze);

  return;
};

/*----------------------------------------------------------------------*
 | evaluate state vector at gauss point                      vuong 09/16 |
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void DRT::ELEMENTS::POROFLUIDMANAGER::VariableManagerInstat<nsd,nen>::EvaluateGPVariables(
    const LINALG::Matrix<nen,1>&   funct,     //! array for shape functions
    const LINALG::Matrix<nsd,nen>& derxy      //! array for shape function derivatives w.r.t x,y,z
    )
{
  // loop over DOFs
  for (int k = 0; k < this->varmanager_->NumDofPerNode(); ++k)
  {
    // history data (or acceleration)
    hist_[k] = funct.Dot(ehist_[k]);
    // calculate time derivative of scalar at t_(n+1)
    phidtnp_[k] = funct.Dot(ephidtnp_[k]);
  }

  // call wrapped class
  this->varmanager_->EvaluateGPVariables(funct,derxy);

  return;
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 | extract node values related to deforming domain           vuong 09/16 |
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void DRT::ELEMENTS::POROFLUIDMANAGER::VariableManagerStruct<nsd,nen>::ExtractElementAndNodeValues(
    const DRT::Element& ele,
    const DRT::Discretization& discretization,
    DRT::Element::LocationArray& la,
    LINALG::Matrix<nsd, nen>& xyze)
{
  // call internal class
  this->varmanager_->ExtractElementAndNodeValues(ele,discretization,la,xyze);

  // determine number of velocity related dofs per node
  const int numveldofpernode = la[ndsvel_].lm_.size()/nen;

  // construct location vector for velocity related dofs
  std::vector<int> lmvel(nsd*nen,-1);
  for (int inode=0; inode<nen; ++inode)
    for (int idim=0; idim<nsd; ++idim)
      lmvel[inode*nsd+idim] = la[ndsvel_].lm_[inode*numveldofpernode+idim];

  // get velocity at nodes
  Teuchos::RCP<const Epetra_Vector> vel = discretization.GetState(ndsvel_, "velocity field");
  if(vel == Teuchos::null)
    dserror("Cannot get state vector velocity");

  // extract local values of velocity field from global state vector
  DRT::UTILS::ExtractMyValues<LINALG::Matrix<nsd,nen> >(*vel,econvelnp_,lmvel);

  // safety check
  Teuchos::RCP<const Epetra_Vector> dispnp = discretization.GetState(ndsdisp_, "dispnp");
  if (dispnp==Teuchos::null)
    dserror("Cannot get state vector 'dispnp'");

  // determine number of displacement related dofs per node
  const int numdispdofpernode = la[ndsdisp_].lm_.size()/nen;

  // construct location vector for displacement related dofs
  std::vector<int> lmdisp(nsd*nen,-1);
  for (int inode=0; inode<nen; ++inode)
    for (int idim=0; idim<nsd; ++idim)
      lmdisp[inode*nsd+idim] = la[ndsdisp_].lm_[inode*numdispdofpernode+idim];

  // extract local values of displacement field from global state vector
  DRT::UTILS::ExtractMyValues<LINALG::Matrix<nsd,nen> >(*dispnp,edispnp_,lmdisp);

  // add nodal displacements to point coordinates
  xyze += edispnp_;

  return;
};

/*----------------------------------------------------------------------*
 | evaluate state vector at gauss point                      vuong 09/16 |
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void DRT::ELEMENTS::POROFLUIDMANAGER::VariableManagerStruct<nsd,nen>::EvaluateGPVariables(
    const LINALG::Matrix<nen,1>&   funct,     //! array for shape functions
    const LINALG::Matrix<nsd,nen>& derxy      //! array for shape function derivatives w.r.t x,y,z
    )
{
  // velocity divergence required for conservative form
  LINALG::Matrix<nsd,nsd> vderxy;
  vderxy.MultiplyNT(econvelnp_,derxy);
  divconvelint_ = 0.0;
  // compute vel x,x  + vel y,y +  vel z,z at integration point
  for (int j = 0; j<nsd; ++j)
    divconvelint_ += vderxy(j,j);

  // convective velocity
  convelint_.Multiply(econvelnp_,funct);

  //gauss point displacements
  dispint_.Multiply(edispnp_,funct);

  // call wrapped class
  this->varmanager_->EvaluateGPVariables(funct,derxy);

  return;
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 | extract node values related to ScaTra coupling           vuong 09/16 |
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void DRT::ELEMENTS::POROFLUIDMANAGER::VariableManagerScalar<nsd,nen>::ExtractElementAndNodeValues(
    const DRT::Element& ele,
    const DRT::Discretization& discretization,
    DRT::Element::LocationArray& la,
    LINALG::Matrix<nsd, nen>& xyze)
{
  // call internal class
  this->varmanager_->ExtractElementAndNodeValues(ele,discretization,la,xyze);

  // get state vector from discretization
  Teuchos::RCP<const Epetra_Vector> scalarnp = discretization.GetState(ndsscalar_, "scalars");
  if (scalarnp==Teuchos::null)
    dserror("Cannot get state vector 'scalars'");

  // determine number of scalars related dofs per node
  const int numscalardofpernode = la[ndsscalar_].lm_.size()/nen;

  // rebuild scalar vector
  escalarnp_.clear();
  escalarnp_.resize(numscalardofpernode,LINALG::Matrix<nen,1>(true));
  // extract local values of displacement field from global state vector
  DRT::UTILS::ExtractMyValues<LINALG::Matrix<nen,1> >(*scalarnp,escalarnp_,la[ndsscalar_].lm_);

  return;
};

/*----------------------------------------------------------------------*
 | evaluate state vector at gauss point                      vuong 09/16 |
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void DRT::ELEMENTS::POROFLUIDMANAGER::VariableManagerScalar<nsd,nen>::EvaluateGPVariables(
    const LINALG::Matrix<nen,1>&   funct,     //! array for shape functions
    const LINALG::Matrix<nsd,nen>& derxy      //! array for shape function derivatives w.r.t x,y,z
    )
{
  // call wrapped class
  this->varmanager_->EvaluateGPVariables(funct,derxy);

  // evaluate scalar values
  if (not escalarnp_.empty())
  {
    scalarnp_.resize(escalarnp_.size(),0.0);
    for (unsigned k = 0; k < escalarnp_.size(); ++k)
      scalarnp_[k] = funct.Dot(escalarnp_[k]);
  }

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
// template classes

// 1D elements
// line 2
template class DRT::ELEMENTS::POROFLUIDMANAGER::VariableManagerInterface<1,2>;

// line 3
template class DRT::ELEMENTS::POROFLUIDMANAGER::VariableManagerInterface<1,3>;

// 2D elements
// tri3
template class DRT::ELEMENTS::POROFLUIDMANAGER::VariableManagerInterface<2,3>;
// quad4
template class DRT::ELEMENTS::POROFLUIDMANAGER::VariableManagerInterface<2,4>;

// quad9 and nurbs9
template class DRT::ELEMENTS::POROFLUIDMANAGER::VariableManagerInterface<2,9>;

// 3D elements
// hex8
template class DRT::ELEMENTS::POROFLUIDMANAGER::VariableManagerInterface<3,8>;

// hex27
template class DRT::ELEMENTS::POROFLUIDMANAGER::VariableManagerInterface<3,27>;
// tet4
template class DRT::ELEMENTS::POROFLUIDMANAGER::VariableManagerInterface<3,4>;
// tet10
template class DRT::ELEMENTS::POROFLUIDMANAGER::VariableManagerInterface<3,10>;
// pyramid5
template class DRT::ELEMENTS::POROFLUIDMANAGER::VariableManagerInterface<3,5>;