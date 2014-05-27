/**
 * Copyright (C) 2013, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	lailong
 * Since	2013-10-9
 */
package org.jabe.neverland.algorithm;

/**
 * “二分插入排序”，利用二分查找优化插入排序中的定位部分。 《算法导论》，习题2.3-6 Observe that the while loop of
 * lines 5 - 7 of the INSERTION-SORT procedure in Section 2.1 uses a linear
 * search to scan (backward) through the sorted subarray A[1  j - 1]. Can we
 * use a binary search (see Exercise 2.3-5) instead to improve the overall
 * worst-case running time of insertion sort to Θ(n lg n)?
 * 本文地址:http://mushiqianmeng.blog.51cto.com/3970029/732333
 * 
 * @Author lailong
 * @Since 2013-10-9
 */
public class InsertSortWithBinarySearch {

	private static int[] input = new int[] { 2, 1, 5, 4, 9, 8, 6, 7, 10, 3 };

	public static void main(String[] args) {
		// 从数组第二个元素开始排序，因为第一个元素本身肯定是已经排好序的
		for (int j = 1; j < input.length; j++) {// 复杂度 n
			// 保存当前值
			int key = input[j];
			// 利用二分查找定位位置
			int index = binarySearch(input, input[j], 0, j - 1);// 复杂度：lgn
			// 将目标插入位置，同时右移目标位置右边的元素
			for (int i = j; i > index; i--) {// 复杂度,最差情况：(n-1)+(n-2)+...+n/2=Θ(n^2)
				input[i] = input[i - 1];
			}
			input[index] = key;
		}
		/*
		 * 复杂度分析： 最佳情况，即都已经排好序，则无需右移，此时时间复杂度为：Θ(n lg n) 最差情况，全部逆序，此时复杂度为Θ(n^2)
		 * 所以针对2.3-6问题，无法将最差情况的复杂度提升到Θ(n lg n)。
		 */
		// 打印数组
		printArray();
	}

	/**
	 * 二分查找
	 * 
	 * @param input
	 *            给定已排序的待查数组
	 * @param target
	 *            查找目标
	 * @param from
	 *            当前查找的范围起点
	 * @param to
	 *            当前查找的返回终点
	 * @return 返回目标在数组中，按顺序应在的位置
	 */
	private static int binarySearch(int[] input, int target, int from, int to) {
		int range = to - from;
		// 如果范围大于0，即存在两个以上的元素，则继续拆分
		if (range > 0) {
			// 选定中间位
			int mid = (to + from) / 2;
			// 如果临界位不满足，则继续二分查找
			if (input[mid] > target) {
				return binarySearch(input, target, from, mid - 1);
			} else {
				return binarySearch(input, target, mid + 1, to);
			}
		} else {
			if (input[from] > target) {
				return from;
			} else {
				return from + 1;
			}
		}
	}

	private static void printArray() {
		for (int i : input) {
			System.out.print(i + " ");
		}
	}

}
