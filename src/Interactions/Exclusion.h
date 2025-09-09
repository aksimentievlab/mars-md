#pragma once

#include "Backend/Resource.h"
#include "Bonded/Bond_IO.h"
#include "Header.h"
#include "System/SimSystem.h"
#include "Types/Types.h"

namespace ARBD {
using int2 = BackendTypes::simd_int2;

class Exclude {
  public:
	Exclude() : ind1(-1), ind2(-1) {}
	Exclude(int ind1, int ind2) : ind1(ind1), ind2(ind2) {}
	bool operator==(const Exclude& e) const;
	bool operator!=(const Exclude& e) const;
	void print();
	int ind1;
	int ind2;
};

class Node {
  public:
	Node(int index);
	void clearTree();
	int makeTree(Node** particles, Bond* bonds, int2* bondMap, int bondstart, int bondend);
	void add(Node* n);
	bool inTree;
	int index;
	int cap;
	int numBonds;
	Node** bonds;
};

// makeExcludes(Bond* bonds, int* bondMap, int num, int numBonds, String exList)
// @param    list of sorted cell bonds; corresponding bond map; number of particles; number of
// bonds;
//           string formated like so "EXCLUDE 1-2 1-3 1-4"; number of excludes
// @return   Array of Excludes
// This algorithm finds the central particle in every bond tree,
// then creates a list of exclusions for the particle pairs
// defined in exList. For example, 1-2 means that there should
// be an exclusion between the central particle and every
// particle it is directly bonded to. 1-3 means that there should
// be an exclusion between the central particle and every particle
// it is two bonds away from
Exclude*
makeExcludes(Bond* bonds, int2* bondMap, int num, int numBonds, String exList, int& numExcludes);
void getExcludes(int root,
				 Node* curr,
				 Exclude* result,
				 int depth,
				 int& capacity,
				 int& numExcludes,
				 bool sentinel,
				 bool* done);

Exclude*
makeExcludes(Bond* bonds, int2* bondMap, int num, int numBonds, String exList, int& numExcludes) {
	int oldNumExcludes = numExcludes;
	numExcludes = 0;
	int resCap = numBonds;
	Exclude* result = new Exclude[numBonds];
	Node** particles = new Node*[num]; // an array of linked lists
	Node** trees;
	int cap = 16;
	trees = new Node*[cap];
	int numTrees = 0;
	for (int i = 0; i < num; i++)
		particles[i] = new Node(i);
	for (int i = 0; i < num; i++) {
		if (!particles[i]->inTree) {
			if (numTrees >= cap) {
				Node** temp = trees;
				cap *= 2;
				trees = new Node*[cap];
				for (int j = 0; j < numTrees; j++)
					trees[j] = temp[j];
				delete temp;
			}

			int bondstart = bondMap[i].x;
			int bondend = bondMap[i].y;
			int nextsize;
			nextsize = particles[i]->makeTree(particles, bonds, bondMap, bondstart, bondend);
			if (nextsize > 1)
				trees[numTrees++] = particles[i];
		}
	}
	printf("exList %s\n", exList.val());
	int depth = atoi(exList.val());

	Node** newTree;
	int treeCap = 100;
	int numNodes = 0;
	for (int i = 0; i < num; i++) {
		Node* p = particles[i];
		if (p->numBonds < 1)
			continue;
		newTree = new Node*[treeCap];
		for (int j = 0; j < num; j++)
			particles[j]->inTree = false;
		newTree[0] = p;
		numNodes = 1;
		int oldNumNodes = 0;
		for (int j = 0; j < depth; j++) {
			int tempNum = numNodes;
			for (int k = oldNumNodes; k < tempNum; k++) {
				oldNumNodes = numNodes;
				Node* p2 = particles[newTree[k]->index];
				p2->inTree = true;
				for (int m = 0; m < p2->numBonds; m++) {
					Node* p3 = p2->bonds[m];
					if (!p3->inTree) {
						p3->inTree = true;
						if (numExcludes >= resCap) {
							printf("Expanding result\n");
							Exclude* tempResult = result;
							resCap *= 2;
							result = new Exclude[resCap];
							for (int n = 0; n < numExcludes; n++)
								result[n] = tempResult[n];
							delete tempResult;
						}
						Exclude ex(i, p3->index);
						result[numExcludes++] = ex;

						if (numNodes >= treeCap) {
							printf("Expanding newTree\n");
							Node** tempTree = newTree;
							treeCap *= 2;
							newTree = new Node*[treeCap];
							for (int n = 0; n < numNodes; n++)
								newTree[n] = tempTree[n];
							delete tempTree;
						}
						newTree[numNodes++] = p3;
					}
				}
			}
		}
		delete[] newTree;
	}

	delete[] particles;
	delete[] trees;
	numExcludes += oldNumExcludes;
	return result;
}

void Exclude::print() {
	printf("EXCLUDE %d %d\n", ind1, ind2);
}

bool Exclude::operator==(const Exclude& e) const {
	return (ind1 == e.ind1) && (ind2 == e.ind2);
}

bool Exclude::operator!=(const Exclude& e) const {
	return !(*this == e);
}

//////////////////////////
// Node Implementations //
//////////////////////////

Node::Node(int index) : index(index) {
	inTree = false;
	cap = 4;
	numBonds = 0;
	bonds = new Node*[cap];
}

void Node::clearTree() {
	printf("index %d cleared\n", index);
	inTree = false;
	for (int i = 0; i < numBonds; i++)
		if (bonds[i]->inTree)
			bonds[i]->clearTree();
}

int Node::makeTree(Node** particles, Bond* bonds, int2* bondMap, int bondstart, int bondend) {
	inTree = true;
	int sum = 1;
	for (int i = bondstart; i < bondend; i++)
		add(particles[bonds[i].ind2]);

	for (int i = bondstart; i < bondend; i++) {
		Node* p = particles[bonds[i].ind2];
		if (!p->inTree)
			sum += p->makeTree(particles, bonds, bondMap, bondMap[p->index].x, bondMap[p->index].y);
	}
	return sum;
}

void Node::add(Node* n) {
	if (numBonds >= cap) {
		Node** temp = bonds;
		cap *= 2;
		bonds = new Node*[cap];
		for (int i = 0; i < numBonds; i++)
			bonds[i] = temp[i];
		delete temp;
	}
	bonds[numBonds++] = n;
}
} // namespace ARBD
