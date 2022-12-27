if [[ -n "$TMUX_PANE" ]]; then
	    session_name=$(tmux list-panes -t "$TMUX_PANE" -F '#S' | head -n1)
fi
echo $session_name
