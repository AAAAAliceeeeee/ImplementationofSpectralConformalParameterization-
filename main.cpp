#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Sparse>
#include <iostream>
#include <Spectra/SymGEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>
#include <Spectra/MatOp/SparseCholesky.h>
#include <igl/matlab/matlabinterface.h>

int main(int argc, char* argv[])
{
  // Basics:
  using namespace Eigen;
  using namespace std;
  MatrixXd V;
  MatrixXi F;
  igl::opengl::glfw::Viewer viewer;
  Engine* engine;


  // Set up Printing Format for Debugging
  std::string processmodel = "models/lion.off";

  // Processing Input Mesh:
  igl::readOFF(processmodel, V, F);
  // Alternative:
  //igl::readOBJ("models/trianglesimple.obj", V, F);

  // Find A
  MatrixXi E;
  MatrixXd J, K;
  igl::boundary_facets(F, E, J, K);

  SparseMatrix<double> A(2 * V.rows(), 2 * V.rows());         // default is column major
  typedef Eigen::Triplet<double> Ta;
  std::vector<Ta> tripletLista;
  tripletLista.reserve(E.rows() * 2);
  for (size_t i = 0; i < E.rows(); i++)
  {
      int u1 = 2 * E.coeffRef(i, 0);
      int u2 = 2 * E.coeffRef(i, 1);
      int v1 = u1 + 1;
      int v2 = u2 + 1;
      tripletLista.push_back(Ta(u1, v2, 0.5));
      tripletLista.push_back(Ta(v2, u1, 0.5));
      tripletLista.push_back(Ta(v1, u2, -0.5));
      tripletLista.push_back(Ta(u2, v1, -0.5));

  }
  A.setFromTriplets(tripletLista.begin(), tripletLista.end());

  //3. Find Ld
  // 1. Set up matrix
  SparseMatrix<double> Ld(2 * V.rows(), 2 * V.rows());
  
  // 2. Loop over the triangles and calculate them 
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  tripletList.reserve(F.rows() * 24);
  

  for (int triangle = 0; triangle < F.rows(); triangle++)
  {
      // 0,1; 0,2; 1,2
      for (int v1 = 0; v1 <= 1; v1++)
      {
          for (int v2 = 2; v2 > v1; v2--)
          {
              int v3 = 3 - v2 - v1;
              int i = F.coeff(triangle, v1);
              int j = F.coeff(triangle, v2);
              int k = F.coeff(triangle, v3);

              // change to uv matrix:
              int u_1 = 2 * i;
              int v_1 = u_1 + 1;
              int u_2 = 2 * j;
              int v_2 = u_2 + 1;

              Vector3d ki = V.row(i) - V.row(k);
              Vector3d kj = V.row(j) - V.row(k);
              double cosikj = (ki.dot(kj)) / (ki.norm() + kj.norm());
              double sinikj = (ki.cross(kj)).norm() / (ki.norm() + kj.norm());
              double cot = cosikj / sinikj;

              tripletList.push_back(T(u_1, u_2, -cot / 2));
              tripletList.push_back(T(u_2, u_1, -cot / 2));
              tripletList.push_back(T(u_1, u_1, cot / 2));
              tripletList.push_back(T(u_2, u_2, cot / 2));


              tripletList.push_back(T(v_1, v_2, -cot / 2));
              tripletList.push_back(T(v_2, v_1, -cot / 2));
              tripletList.push_back(T(v_1, v_1, cot / 2));
              tripletList.push_back(T(v_2, v_2, cot / 2));
            
          }
      }
  }
  Ld.setFromTriplets(tripletList.begin(), tripletList.end());

  //4. Find Lc
  SparseMatrix<double> Lc(2 * V.rows(), 2 * V.rows());
  Lc = Ld - A;
  //5. Minimize Lc by eigs 
  // find generalivzed eigen vector: Lc*u = s*B*u // B=A or //
  //5.1: B = A

  //EigsType EIGS_TYPE_SM;
  SparseMatrix<double> id(A.rows(), A.rows());
  id.setIdentity();

  //5.2 B="diagonal matrix with 1 at each diagonal element corresponding to boundary vertices)
  SparseMatrix<double> B(A.rows(), A.rows());
  typedef Eigen::Triplet<double> Tb;
  std::vector<Tb> tripletListb;
  tripletListb.reserve(E.rows() * 4);
  for (size_t i = 0; i < E.rows(); i++)
  {
      int u1 = 2 * E.coeffRef(i, 0);
      int u2 = 2 * E.coeffRef(i, 1);
      int v1 = u1 + 1;
      int v2 = u2 + 1;
      tripletListb.push_back(T(u1, u1, 0.5));
      tripletListb.push_back(T(u2, u2, 0.5));
      tripletListb.push_back(T(v1, v1, 0.5));
      tripletListb.push_back(T(v2, v2, 0.5));
  }

  B.setFromTriplets(tripletListb.begin(), tripletListb.end());
  B.setIdentity();

  // Setting up to find the largest eigen value
  // [B-1/Vb eb eb.T]u = miu Lc U
  //1. Set up Matrix e
  SparseMatrix<double> e (2 * V.rows(), 2);
  typedef Eigen::Triplet<double> Te;
  std::vector<Te> tripletListe;
  tripletListe.reserve(V.rows() * 2);
  for (size_t i = 0; i < V.rows(); i++)
  {
      tripletListe.push_back(T(i, 0, 1));
      tripletListe.push_back(T(V.rows() + i, 1, 1));
  }

  e.setFromTriplets(tripletListe.begin(), tripletListe.end());

  
  SparseMatrix<double> eb = B * e;
  double Vb = E.rows(); // Is this Correct?
  //SparseMatrix<double> LHS = (B - 1/Vb * eb * eb.transpose());
  cout << "all matrix constructed" << endl;

  // using matlab to calculate: (Method 1)
  MatrixXd sVectors;
  MatrixXd sValues;
  igl::matlab::mlinit(&engine);
  igl::matlab::mlsetmatrix(&engine,"Lc",Lc);
  igl::matlab::mlsetmatrix(&engine,"A",A);
  igl::matlab::mlsetmatrix(&engine,"B",B);
  igl::matlab::mlsetmatrix(&engine,"E",E);
  igl::matlab::mlsetmatrix(&engine, "e", e);
  igl::matlab::mleval(&engine,"[V,D] = eigs(Lc,A,5,'smallestabs')");
  igl::matlab::mlgetmatrix(&engine,"V",sVectors);
  igl::matlab::mlgetmatrix(&engine,"D",sValues);
  
  cout << "Method 1 Solution: " << endl;
 // cout << "V: " << endl;
 // cout << sVectors << endl;
  cout << "D" << endl;
  cout << sValues << endl;

  // (Method 2)
  MatrixXd sVectors2;
  MatrixXd sValues2;
  igl::matlab::mlsetmatrix(&engine,"eb", eb);
  cout << "1" << endl;
  igl::matlab::mlsetscalar(&engine, "Vb", Vb);
  cout << "2" << endl;
  //igl::matlab::mleval(&engine, "LHS = B - (1/Vb * eb) * eb.transpose()");
  igl::matlab::mleval(&engine, "[V2,D2] = eigs((B - (1/Vb * eb) * eb'), Lc, 5, 'largestabs')");
  cout << "3" << endl;
  igl::matlab::mlgetmatrix(&engine,"V2", sVectors2);
  igl::matlab::mlgetmatrix(&engine,"D2",sValues2);
  cout << "Method 2 Solution: " << endl;
 // cout << "V:" << endl;
 // cout << sVectors2 << endl;
  cout << "D: " << endl;
  cout << sValues2 << endl;


  bool Found = false;
  int it = 0; 
  // check result: 
  VectorXd uA;
  while (!Found && it < 5)
  {
    cout << "i = "<< it << endl;
    VectorXd u = sVectors2.col(it);
    MatrixXd R1 = (u.transpose() * B) * e;
    cout << "R1: " << endl;
    cout << R1.sum() << endl; 
    MatrixXd R2 = (u.transpose() * B) * u;
    cout << "R2: " << endl; 
    cout << R2 << endl; 

    if (R2.isApprox(MatrixXd::Identity(1, 1)) && R1.sum() <= 0.0005 && R1.sum() >= -0.0005)
    {
      Found = true;
      uA = sVectors2.col(it);
      cout << "FOUND!!!!!" << endl; 
      break;
    }
    it ++; 
  }
  cout << "All values evaluated" << endl;

  if (!Found)
  {
    uA = sVectors2.col(2);
    cout << "Defaulted to col 2" << endl;
  }
  // Output Image  
  MatrixXd V2(V.rows(),3);
  for (int k = 0; k < V.rows(); k++)
  {
      // Draw Points (temp)
      //viewer.data().add_points(Eigen::RowVector3d(uA.coeffRef(2*k), uA(2*k + 1), 0), Eigen::RowVector3d(1, 0, 0));
      //MatrixXd V2(V.rows(),V.cols());
      //cout << "k: " << k << endl;
      V2.coeffRef(k, 0) = uA(2 * k);
      V2.coeffRef(k, 1) = uA(2 * k + 1);
      V2.coeffRef(k, 2) = 0;
  }
 
    

  viewer.data().set_mesh(V2, F);
  viewer.data().show_lines = true;
  viewer.launch();

}