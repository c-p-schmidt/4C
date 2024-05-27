/*----------------------------------------------------------------------------*/
/*! \file
\brief 2D wall element for structure part of porous medium including scatra functionality.

\level 2


*/
/*---------------------------------------------------------------------------*/

#ifndef FOUR_C_W1_PORO_SCATRA_HPP
#define FOUR_C_W1_PORO_SCATRA_HPP

#include "4C_config.hpp"

#include "4C_inpar_scatra.hpp"
#include "4C_w1_poro.hpp"
#include "4C_w1_poro_scatra_eletypes.hpp"

FOUR_C_NAMESPACE_OPEN

namespace DRT
{
  // forward declarations
  class Discretization;

  namespace ELEMENTS
  {
    /*!
    \brief A C++ version of a 2 dimensional solid element with modifications for porous media

    */
    template <CORE::FE::CellType distype>
    class Wall1PoroScatra : public Wall1Poro<distype>
    {
      typedef DRT::ELEMENTS::Wall1Poro<distype> my;

     public:
      //@}
      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor

      \param id : A unique global id
      \param owner : elements owner
      */
      Wall1PoroScatra(int id, int owner);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      Wall1PoroScatra(const Wall1PoroScatra& old);

      //@}

      /*!
      \brief Deep copy this instance of Wall1_Poro_Scatra and return pointer to the copy

      The Clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed
      */
      DRT::Element* Clone() const override;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of this file.
      */
      int UniqueParObjectId() const override
      {
        int parobjectid(-1);
        switch (distype)
        {
          case CORE::FE::CellType::tri3:
          {
            parobjectid = DRT::ELEMENTS::WallTri3PoroScatraType::Instance().UniqueParObjectId();
            break;
          }
          case CORE::FE::CellType::quad4:
          {
            parobjectid = DRT::ELEMENTS::WallQuad4PoroScatraType::Instance().UniqueParObjectId();
            break;
          }
          case CORE::FE::CellType::quad9:
          {
            parobjectid = DRT::ELEMENTS::WallQuad9PoroScatraType::Instance().UniqueParObjectId();
            break;
          }
          case CORE::FE::CellType::nurbs4:
          {
            parobjectid = DRT::ELEMENTS::WallNurbs4PoroScatraType::Instance().UniqueParObjectId();
            break;
          }
          case CORE::FE::CellType::nurbs9:
          {
            parobjectid = DRT::ELEMENTS::WallNurbs9PoroScatraType::Instance().UniqueParObjectId();
            break;
          }
          default:
          {
            FOUR_C_THROW("unknown element type");
            break;
          }
        }
        return parobjectid;
      };

      /*!
      \brief Pack this class so it can be communicated

      \ref Pack and \ref Unpack are used to communicate this element
      */
      void Pack(CORE::COMM::PackBuffer& data) const override;

      /*!
      \brief Unpack data from a char vector into this class

      \ref Pack and \ref Unpack are used to communicate this element
      */
      void Unpack(const std::vector<char>& data) override;

      //! @name Access methods

      /*!
      \brief Print this element
      */
      void Print(std::ostream& os) const override;

      /*!
      \brief Return elementtype instance
      */
      DRT::ElementType& ElementType() const override
      {
        switch (distype)
        {
          case CORE::FE::CellType::tri3:
            return DRT::ELEMENTS::WallTri3PoroScatraType::Instance();
            break;
          case CORE::FE::CellType::quad4:
            return DRT::ELEMENTS::WallQuad4PoroScatraType::Instance();
            break;
          case CORE::FE::CellType::quad9:
            return DRT::ELEMENTS::WallQuad9PoroScatraType::Instance();
            break;
          case CORE::FE::CellType::nurbs4:
            return DRT::ELEMENTS::WallNurbs4PoroScatraType::Instance();
            break;
          case CORE::FE::CellType::nurbs9:
            return DRT::ELEMENTS::WallNurbs9PoroScatraType::Instance();
            break;
          default:
            FOUR_C_THROW("unknown element type");
            break;
        }
        return DRT::ELEMENTS::WallQuad4PoroScatraType::Instance();
      };

      //@}

      //! @name Input and Creation

      /*!
      \brief Read input for this element
      */
      bool ReadElement(const std::string& eletype, const std::string& eledistype,
          INPUT::LineDefinition* linedef) override;

      //@}

      /// @name params
      /*!
      \brief Return the SCATRA ImplType
      */
      const INPAR::SCATRA::ImplType& ImplType() const { return impltype_; };

     private:
      //! implementation type (physics)
      INPAR::SCATRA::ImplType impltype_;
      //@}

     protected:
      //! don't want = operator
      Wall1PoroScatra& operator=(const Wall1PoroScatra& old);

    };  // class Wall1_Poro_Scatra

  }  // namespace ELEMENTS
}  // namespace DRT
FOUR_C_NAMESPACE_CLOSE

#endif