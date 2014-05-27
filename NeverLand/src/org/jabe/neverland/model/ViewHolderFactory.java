package org.jabe.neverland.model;

import org.jabe.neverland.R;

import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

public class ViewHolderFactory {
	public static class DragListViewHolder extends ViewHolder implements IViewHolder {
		TextView title;
		TextView content;
		ViewGroup box1;
		ViewGroup box2;
		@Override
		public void saveView(View view, int position, boolean save) {
			title = (TextView) view.findViewById(R.id.drag_list_item_text_title);
			content = (TextView) view.findViewById(R.id.drag_list_item_text_content);
			box1 = (ViewGroup) view.findViewById(R.id.box1);
			box2 = (ViewGroup) view.findViewById(R.id.box2);
			save(view, this, save);
		}
	}
	public static class SimpleListItem extends ViewHolder implements IViewHolder {
		TextView text1;
		@Override
		public void saveView(View view, int position, boolean save) {
			text1 = (TextView) view.findViewById(android.R.id.text1);
			save(view, this, save);
		}
		
	}
}
