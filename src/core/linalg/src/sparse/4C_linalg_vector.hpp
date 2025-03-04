// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_VECTOR_HPP
#define FOUR_C_LINALG_VECTOR_HPP


#include "4C_config.hpp"

#include "4C_linalg_multi_vector.hpp"

#include <Epetra_FEVector.h>
#include <Epetra_IntVector.h>
#include <Epetra_Vector.h>

#include <memory>

// Do not lint the file for identifier names, since the naming of the Wrapper functions follow the
// naming of the Epetra_Vector

// NOLINTBEGIN(readability-identifier-naming)

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  template <typename T>
  class VectorView;

  // Sparse Vector which will replace the Epetra_Vector
  template <typename T>
  class Vector
  {
    static_assert(std::is_same_v<T, double>, "Only double is supported for now");

   public:
    /// Basic vector constructor to create vector based on a map and initialize memory with zeros
    explicit Vector(const Epetra_BlockMap& Map, bool zeroOut = true);

    /// Copy constructor from epetra to vector
    explicit Vector(const Epetra_Vector& Source);

    explicit Vector(const Epetra_FEVector& Source);


    // Rule of five: We currently need to take care to make a deep copy of the Epetra_Vector.
    Vector(const Vector& other);

    Vector& operator=(const Vector& other);

    ~Vector() = default;

    // Implicit conversion to MultiVector: the MultiVector will view the same content and only have
    // a single column.
    operator const MultiVector<T>&() const;
    operator MultiVector<T>&();

    // (Implicit) conversions: they all return references or RCPs, never copies
    const Epetra_Vector& get_ref_of_Epetra_Vector() const { return *vector_; }

    Epetra_Vector& get_ref_of_Epetra_Vector() { return *vector_; }

    std::shared_ptr<Epetra_Vector> get_ptr_of_Epetra_Vector() { return vector_; }

    operator Epetra_MultiVector&() { return *vector_; }

    operator const Epetra_MultiVector&() const { return *vector_; }

    operator Epetra_Vector&() { return *vector_; }

    operator const Epetra_Vector&() const { return *vector_; }

    //! get pointer of epetra multi vector
    std::shared_ptr<Epetra_MultiVector> get_ptr_of_Epetra_MultiVector() { return vector_; }

    //! Temporary helper to ease transition from Epetra and simplify interfacing with RCP-laden code
    std::shared_ptr<MultiVector<T>> get_ptr_of_MultiVector() const
    {
      sync_view();
      return multi_vector_view_;
    }

    //! Computes dot product of each corresponding pair of vectors.
    int Dot(const Epetra_MultiVector& A, double* Result) const;

    //! Puts element-wise absolute values of input Multi-vector in target.
    int Abs(const Epetra_MultiVector& A);

    //! Replace multi-vector values with scaled values of A, \e this = ScalarA*A.
    int Scale(double ScalarA, const Epetra_MultiVector& A);

    //! Update multi-vector values with scaled values of A, \e this = ScalarThis*\e this +
    //! ScalarA*A.
    int Update(double ScalarA, const Epetra_MultiVector& A, double ScalarThis);

    //! Update multi-vector with scaled values of A and B, \e this = ScalarThis*\e this + ScalarA*A
    //! + ScalarB*B.
    int Update(double ScalarA, const Epetra_MultiVector& A, double ScalarB,
        const Epetra_MultiVector& B, double ScalarThis);


    ///

    //! Compute 1-norm of each vector
    int Norm1(double* Result) const;

    //! Compute 2-norm of each vector
    int Norm2(double* Result) const;

    //! Compute Inf-norm of each vector
    int NormInf(double* Result) const;

    //! Compute minimum value of each vector
    int MinValue(double* Result) const;

    //! Compute maximum value of each vector
    int MaxValue(double* Result) const;

    //! Compute mean (average) value of each vector
    int MeanValue(double* Result) const;

    //! Scale the current values of a multi-vector, \e this = ScalarValue*\e this.
    int Scale(double ScalarValue) { return vector_->Scale(ScalarValue); }

    //! Computes dot product of each corresponding pair of vectors.
    int Dot(const Vector& A, double* Result) const;

    //! Puts element-wise absolute values of input Multi-vector in target.
    int Abs(const Vector& A);

    //! Replace multi-vector values with scaled values of A, \e this = ScalarA*A.
    int Scale(double ScalarA, const Vector& A);

    //! Update multi-vector values with scaled values of A, \e this = ScalarThis*\e this +
    //! ScalarA*A.
    int Update(double ScalarA, const Vector& A, double ScalarThis);

    //! Update multi-vector with scaled values of A and B, \e this = ScalarThis*\e this + ScalarA*A
    //! + ScalarB*B.
    int Update(double ScalarA, const Vector& A, double ScalarB, const Vector& B, double ScalarThis);

    //! Initialize all values in a multi-vector with const value.
    int PutScalar(double ScalarConstant);

    //! Element access function
    double& operator[](int index) { return (*vector_)[index]; }

    double operator[](int const index) const { return (*vector_)[index]; }

    //! Returns the address of the Epetra_BlockMap for this multi-vector.
    const Epetra_BlockMap& Map() const { return (vector_->Map()); };

    //! Returns the MPI_Comm for this multi-vector.
    MPI_Comm Comm() const;

    //! Returns true if this multi-vector is distributed global, i.e., not local replicated.
    bool DistributedGlobal() const { return (vector_->Map().DistributedGlobal()); };

    //! Print method
    void Print(std::ostream& os) const { vector_->Print(os); }

    //! Returns the number of vectors in the multi-vector.
    int NumVectors() const { return vector_->NumVectors(); }

    //! Returns the local vector length on the calling processor of vectors in the multi-vector.
    int MyLength() const { return vector_->MyLength(); }

    //! Returns the global vector length of vectors in the multi-vector.
    int GlobalLength() const { return vector_->GlobalLength(); }

    //! Replace values in a vector with a given indexed list of values, indices are in local index
    //! space.
    int ReplaceMyValues(int NumEntries, const double* Values, const int* Indices)
    {
      return vector_->ReplaceMyValues(NumEntries, Values, Indices);
    }


    //! Replace values in a vector with a given indexed list of values at the specified BlockOffset,
    //! indices are in local index space.
    int ReplaceMyValues(int NumEntries, int BlockOffset, const double* Values, const int* Indices)
    {
      return vector_->ReplaceMyValues(NumEntries, BlockOffset, Values, Indices);
    }

    int ReplaceMyValue(int MyRow, int VectorIndex, double ScalarValue)
    {
      return vector_->ReplaceMyValue(MyRow, VectorIndex, ScalarValue);
    }

    double* Values() const { return vector_->Values(); }

    /** Replace map, only if new map has same point-structure as current map.
        return 0 if map is replaced, -1 if not.
     */
    int ReplaceMap(const Epetra_BlockMap& map);

    int ReplaceGlobalValue(int GlobalRow, int VectorIndex, double ScalarValue)
    {
      return vector_->ReplaceGlobalValue(GlobalRow, VectorIndex, ScalarValue);
    }

    int ReplaceGlobalValue(long long GlobalRow, int VectorIndex, double ScalarValue)
    {
      return vector_->ReplaceGlobalValue(GlobalRow, VectorIndex, ScalarValue);
    }

    //! Matrix-Matrix multiplication, \e this = ScalarThis*\e this + ScalarAB*A*B.
    int Multiply(char TransA, char TransB, double ScalarAB, const Epetra_MultiVector& A,
        const Epetra_MultiVector& B, double ScalarThis)
    {
      return vector_->Multiply(TransA, TransB, ScalarAB, A, B, ScalarThis);
    }

    //! Puts element-wise reciprocal values of input Multi-vector in target.
    int Reciprocal(const Epetra_MultiVector& A) { return vector_->Reciprocal(A); }

    //! Multiply a Core::LinAlg::MultiVector<double> with another, element-by-element.
    int Multiply(double ScalarAB, const Epetra_MultiVector& A, const Epetra_MultiVector& B,
        double ScalarThis)
    {
      return vector_->Multiply(ScalarAB, A, B, ScalarThis);
    }

    int ReplaceGlobalValues(int NumEntries, const double* Values, const int* Indices)
    {
      return vector_->ReplaceGlobalValues(NumEntries, Values, Indices);
    }

    int ReplaceGlobalValues(int NumEntries, const double* Values, const long long* Indices)
    {
      return vector_->ReplaceGlobalValues(NumEntries, Values, Indices);
    }

    //! Imports an Epetra_DistObject using the Epetra_Import object.
    int Import(const Epetra_SrcDistObject& A, const Epetra_Import& Importer,
        Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor = nullptr)
    {
      return vector_->Import(A, Importer, CombineMode, Indexor);
    }

    //! Imports an Epetra_DistObject using the Epetra_Export object.
    int Import(const Epetra_SrcDistObject& A, const Epetra_Export& Exporter,
        Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor = nullptr)
    {
      return vector_->Import(A, Exporter, CombineMode, Indexor);
    }

    int Export(const Epetra_SrcDistObject& A, const Epetra_Import& Importer,
        Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor = nullptr)
    {
      return vector_->Export(A, Importer, CombineMode, Indexor);
    }

    int Export(const Epetra_SrcDistObject& A, const Epetra_Export& Exporter,
        Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor = nullptr)
    {
      return vector_->Export(A, Exporter, CombineMode, Indexor);
    }

    int SumIntoGlobalValue(int GlobalRow, int VectorIndex, double ScalarValue)
    {
      return vector_->SumIntoGlobalValue(GlobalRow, VectorIndex, ScalarValue);
    }

    int SumIntoGlobalValue(long long GlobalRow, int VectorIndex, double ScalarValue)
    {
      return vector_->SumIntoGlobalValue(GlobalRow, VectorIndex, ScalarValue);
    }

    int SumIntoGlobalValues(
        int NumEntries, int BlockOffset, const double* Values, const int* Indices)
    {
      return vector_->SumIntoGlobalValues(NumEntries, BlockOffset, Values, Indices);
    }

    int SumIntoGlobalValues(int NumEntries, const double* Values, const int* Indices)
    {
      return vector_->SumIntoGlobalValues(NumEntries, Values, Indices);
    }

    int ReciprocalMultiply(double ScalarAB, const Epetra_MultiVector& A,
        const Epetra_MultiVector& B, double ScalarThis)
    {
      return vector_->ReciprocalMultiply(ScalarAB, A, B, ScalarThis);
    }

    int SumIntoMyValue(int MyRow, int VectorIndex, double ScalarValue)
    {
      return vector_->SumIntoMyValue(MyRow, VectorIndex, ScalarValue);
    }


    int SumIntoMyValue(int MyBlockRow, int BlockRowOffset, int VectorIndex, double ScalarValue)
    {
      return vector_->SumIntoMyValue(MyBlockRow, BlockRowOffset, VectorIndex, ScalarValue);
    }

    int SumIntoMyValues(int NumEntries, const double* Values, const int* Indices)
    {
      return vector_->SumIntoMyValues(NumEntries, Values, Indices);
    }

   private:
    Vector() = default;

    /**
     * This function ensures the view necessary to obtain MultiVector object is in sync.
     * Internally, it will be only called once. However, it is important that this call is delayed
     * until a view is actually required. If views were constructed ahead of time, we could run into
     * an infinite recursion between Vector and MultiVector.
     */
    void sync_view() const;

    /**
     * Special constructor useful for converting a column of our MultiVector to our Vector.
     * @param view The internals that this Vector should view.
     */
    [[nodiscard]] static std::shared_ptr<Vector<T>> create_view(const Epetra_Vector& view)
    {
      std::shared_ptr<Vector<T>> ret(new Vector<T>);
      ret->vector_ = std::make_shared<Epetra_Vector>(Epetra_DataAccess::View, view, 0);
      return ret;
    }

    //! The actual Epetra_Vector object.
    std::shared_ptr<Epetra_Vector> vector_;
    //! MultiVector view of the Vector. This is used to allow implicit conversion to MultiVector.
    mutable std::shared_ptr<MultiVector<T>> multi_vector_view_;

    friend class VectorView<Vector<T>>;
    friend class VectorView<const Vector<T>>;
    friend class MultiVector<T>;
  };

  /**
   * Specialization of the Vector class for int.
   *
   * @note Currently, this specialization is mandated by a separate implementation of
   * Epetra_IntVector.
   */
  template <>
  class Vector<int>
  {
   public:
    explicit Vector(const Epetra_BlockMap& map, bool zeroOut = true);

    Vector(const Epetra_BlockMap& map, int* values);

    Vector(const Vector& other);
    Vector& operator=(const Vector& other);
    Vector(Vector&& other) noexcept;
    Vector& operator=(Vector&& other) noexcept;

    int PutValue(int Value);

    int MaxValue();

    int MinValue();

    int& operator[](int index) { return (*vector_)[index]; }

    const int& operator[](int index) const { return (*vector_)[index]; }

    int* Values() const { return vector_->Values(); };

    int MyLength() const { return vector_->MyLength(); };

    int GlobalLength() const { return vector_->GlobalLength(); };

    void Print(std::ostream& os) const;

    const Epetra_BlockMap& Map() const { return (vector_->Map()); };

    //! Imports an Epetra_DistObject using the Epetra_Import object.
    int Import(const Vector& A, const Epetra_Import& Importer, Epetra_CombineMode CombineMode,
        const Epetra_OffsetIndex* Indexor = nullptr)
    {
      return vector_->Import(*A.vector_, Importer, CombineMode, Indexor);
    }

    //! Imports an Epetra_DistObject using the Epetra_Export object.
    int Import(const Vector& A, const Epetra_Export& Exporter, Epetra_CombineMode CombineMode,
        const Epetra_OffsetIndex* Indexor = nullptr)
    {
      return vector_->Import(*A.vector_, Exporter, CombineMode, Indexor);
    }

    int Export(const Vector& A, const Epetra_Import& Importer, Epetra_CombineMode CombineMode,
        const Epetra_OffsetIndex* Indexor = nullptr)
    {
      return vector_->Export(*A.vector_, Importer, CombineMode, Indexor);
    }

    int Export(const Vector& A, const Epetra_Export& Exporter, Epetra_CombineMode CombineMode,
        const Epetra_OffsetIndex* Indexor = nullptr)
    {
      return vector_->Export(*A.vector_, Exporter, CombineMode, Indexor);
    }

    [[nodiscard]] MPI_Comm Comm() const;

   private:
    std::shared_ptr<Epetra_IntVector> vector_;
  };

  /**
   * Temporary helper class for migration. View one of the Trilinos vector types as one of ours.
   * Make sure that the viewed vector lives longer than the view.
   */
  template <typename VectorType>
  class VectorView
  {
   public:
    template <typename EpetraVectorType>
    VectorView(EpetraVectorType& vector) : view_vector_(VectorType::create_view(vector))
    {
    }

    // Make the class hard to misuse and disallow copy and move.
    VectorView(const VectorView& other) = delete;
    VectorView& operator=(const VectorView& other) = delete;
    VectorView(VectorView&& other) = delete;
    VectorView& operator=(VectorView&& other) = delete;
    ~VectorView() = default;

    //! Allow implicit conversion to Vector for use in new interfaces.
    operator VectorType&() { return *view_vector_; }

    //! Allow implicit conversion to RCP<Vector> for use in new interfaces.
    std::shared_ptr<VectorType>& get_non_owning_rcp_ref() { return view_vector_; }

   private:
    //! Store inside an RCP because our interfaces frequently take references to RCPs.
    std::shared_ptr<VectorType> view_vector_;
  };


  // Template deduction guide for view of Epetra_Vector
  VectorView(Epetra_Vector&) -> VectorView<Vector<double>>;
  VectorView(const Epetra_Vector&) -> VectorView<const Vector<double>>;

  // Template deduction guide for view of Epetra_MultiVector
  VectorView(Epetra_MultiVector&) -> VectorView<MultiVector<double>>;
  VectorView(const Epetra_MultiVector&) -> VectorView<const MultiVector<double>>;

  // Template deduction guide for view of Epetra_FEVector
  VectorView(Epetra_FEVector&) -> VectorView<MultiVector<double>>;
  VectorView(const Epetra_FEVector&) -> VectorView<const MultiVector<double>>;

}  // namespace Core::LinAlg



FOUR_C_NAMESPACE_CLOSE

// NOLINTEND(readability-identifier-naming)

#endif